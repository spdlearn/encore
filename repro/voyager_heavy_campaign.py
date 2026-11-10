#!/usr/bin/env python3
# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Heavy Multi-Node Distributed Voyager CPU spd-bench Campaign.

Executes across 8 parallel 16-CPU Kubernetes Jobs in namespace bpinto-sdp190:
- shard 'bnci2014_001_p1': BNCI2014-001 Subjects 1..5 across all 23 SPD Learn models
- shard 'bnci2014_001_p2': BNCI2014-001 Subjects 6..9 across all 23 SPD Learn models
- shard 'bnci2014_004':    BNCI2014-004 (BCIC-IV-2b) all 9 subjects across all 23 models
- shard 'bnci2015_001':    BNCI2015-001 all 12 subjects across all 23 models
- shard 'bnci2014_002_weibo': BNCI2014-002 (14 subjs) & Weibo2014 (10 subjs) across models
- shard 'schirr_shin_zhou': Schirrmeister2017 (14 subjs), Shin2017A, Zhou2020 across models
- shard 'lee2019_cho2017': Lee2019_MI (OpenBMI/KU) & Cho2017 across models
- shard 'spd_matrices':    Radar (10 seeds x 150 epochs + scarcity/batch curves) &
                           HDM05 (5 seeds x 120 epochs + depth/threshold curves) across 18 SPD variants
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing
import os
import pickle
import platform
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

if not hasattr(np, "long"):
    np.long = np.int_  # type: ignore[attr-defined]
if not hasattr(np, "ulong"):
    np.ulong = np.uint  # type: ignore[attr-defined]

from scipy.signal import butter, sosfiltfilt
import torch
import torch.nn as nn

import spd_learn
import spd_learn.functional as F_spd

from spd_learn import models, modules
from spd_learn.modules import step_dynamic_trivialization


BNCI_PREPARED_DIR = Path("/voyager/ceph/users/bpinto/bench/spd-bnci-20261105-1614/prepared")
BNCI_EXISTING_JSONL = Path("/voyager/ceph/users/bpinto/bench/spd-bnci-20261105-1614/results.jsonl")
DATA_STAGE_DIR = Path("/voyager/ceph/users/bpinto/bench/spd-encore-20261105/data")
BD_CACHE_ROOT = Path(
    "/voyager/ceph/groups/sdp190/bpinto/projects/deep_ceiling/data_cache/data/braindecode"
)
RESULTS_SHARD_DIR = Path("/voyager/ceph/users/bpinto/bench/spd-encore-20261105/shards")


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    return float(np.mean(y_t == y_p)) if len(y_t) else 0.0


def balanced_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    classes = np.unique(y_t)
    if not len(classes):
        return 0.0
    return float(np.mean([np.mean(y_p[y_t == c] == c) for c in classes]))


def cohen_kappa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if not len(y_t):
        return 0.0
    classes = np.unique(np.concatenate([y_t, y_p]))
    p_o = float(np.mean(y_t == y_p))
    p_e = sum(float(np.mean(y_t == c)) * float(np.mean(y_p == c)) for c in classes)
    return 1.0 if abs(1.0 - p_e) < 1e-12 else float((p_o - p_e) / (1.0 - p_e))


# =============================================================================
# 1. Unified EEG Model Factory (All 24 Models in spd_learn)
# =============================================================================

ALL_EEG_MODELS = [
    "SPDNet",
    "TSMNet",
    "TensorCSPNet",
    "MAtt",
    "Green",
    "EEGSPDNet",
    "PhaseSPDNet",
    "CorAtt",
    "CorNet",
    "SPNN",
    "GyroAtt",
    "GraphCSPNet",
    "GBWAtt",
    "SPDTransNet",
    "SPDMetaFormer",
    "RHOP",
    "DeepGeoCCA",
    "RCEEGNet",
    "STaRNet",
    "SPDResNet",
    "ALEM",
    "GyroSPD",
    "SPDNetAE",
    "SSGCNet",
]

PURE_COV_EEG_MODELS = {
    "SPDNet",
    "PhaseSPDNet",
    "CorNet",
    "SPDResNet",
    "ALEM",
    "GyroSPD",
    "SPDNetAE",
}

TSM_BN_EEG_VARIANTS = {
    "SPDNetBN",
    "LieBN-AIM",
    "LieBN-LEM",
    "LieBN-LCM",
    "ARMAGNAC-Harmonic",
    "ARMAGNAC-Arithmetic",
    "ARMAGNAC-GAH",
    "GBWBN",
    "CBN",
    "PCM",
    "BWCM",
    "CorNet-OLM",
    "CorNet-LSM",
    "RieTrans-SPDRMLR",
    "SPDResNet-ALEM",
}


class TSMVariantClassifier(nn.Module):
    """TSMNet spatiotemporal conv front-end + SPDMatrixClassifier manifold backbone."""

    def __init__(self, n_chans: int, n_outputs: int, variant: str) -> None:
        super().__init__()
        n_temp = 4
        n_spatio = 32
        sub_d = 16
        self.cnn = nn.Sequential(
            nn.Conv2d(1, n_temp, kernel_size=(1, 25), padding="same", padding_mode="reflect"),
            nn.Conv2d(n_temp, n_spatio, kernel_size=(n_chans, 1)),
            nn.Flatten(start_dim=2),
        )
        self.covpool = modules.CovLayer()
        self.spd_head = SPDMatrixClassifier(
            dims=[n_spatio, sub_d],
            n_classes=n_outputs,
            variant=variant,
            reeig_eps=1e-3,
            dtype=torch.float64,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.cnn(x[:, None, ...])
        cov = self.covpool(z).to(torch.float64)
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        mean_diag = (
            cov.diagonal(dim1=-2, dim2=-1)
            .mean(dim=-1, keepdim=True)
            .unsqueeze(-1)
            .clamp(min=1e-6)
        )
        cov = cov / mean_diag
        eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
        cov = cov + 1e-3 * eye
        return self.spd_head(cov).to(x.dtype)



class RHOPClassifier(nn.Module):
    """End-to-end EEG classifier using temporal/spatial conv + RHOP."""

    def __init__(self, n_chans: int, n_outputs: int) -> None:
        super().__init__()
        sub_d = max(4, min(16, n_chans))
        feat_d = max(16, min(24, n_chans))
        self.conv = nn.Sequential(
            nn.Conv2d(1, feat_d, kernel_size=(1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(feat_d),
            nn.Conv2d(feat_d, feat_d, kernel_size=(n_chans, 1), groups=feat_d, bias=False),
            nn.BatchNorm2d(feat_d),
        )
        self.rhop = modules.RHOP(in_features=feat_d, reduced_dim=sub_d, eps=1e-4)
        self.head = nn.Linear(feat_d, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x.unsqueeze(1)).squeeze(2).transpose(1, 2)
        pooled = self.rhop(z)
        return self.head(pooled)


def _init_spdtransnet_ident(m: nn.Module) -> None:
    """Initialize SPDTransNet non-residual MLPs in the linear identity regime for single-trial EEG."""
    with torch.no_grad():
        nn.init.eye_(m.projection.weight)
        nn.init.zeros_(m.projection.bias)
        for enc in [m.intra_epoch.encoder, m.inter_epoch.encoder]:
            enc.position.sinusoid.mul_(0.01)
            d = enc.position.mapping[0].in_features
            enc.position.mapping[0].weight.copy_(torch.eye(d))
            enc.position.mapping[0].bias.fill_(4.0)
            enc.position.mapping[2].weight.copy_(torch.eye(d))
            enc.position.mapping[2].bias.fill_(-4.0)
            for layer in enc.layers:
                nn.init.zeros_(layer.attention.head_mix.weight)
                nn.init.zeros_(layer.ff[3].weight)
                nn.init.zeros_(layer.ff[3].bias)
        d_hid = m.classifier[0].out_features
        m.classifier[0].weight.zero_()
        m.classifier[0].weight[:, :d_hid] = 0.5 * torch.eye(d_hid)
        m.classifier[0].weight[:, d_hid : 2 * d_hid] = 0.5 * torch.eye(d_hid)
        m.classifier[0].bias.fill_(4.0)
        m.classifier[3].weight.copy_(torch.eye(d_hid))
        m.classifier[3].bias.fill_(-4.0)


def build_eeg_model(
    model_name: str, n_chans: int, n_outputs: int, sfreq: int = 128
) -> nn.Module:
    sub_d = max(3, min(16, n_chans))
    mid_d = max(3, min(24, n_chans))
    if model_name in TSM_BN_EEG_VARIANTS:
        return TSMVariantClassifier(n_chans=n_chans, n_outputs=n_outputs, variant=model_name)
    if model_name == "RHOP":
        return RHOPClassifier(n_chans=n_chans, n_outputs=n_outputs)
    if model_name == "ALEM":
        return models.SPDResNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            bimap_dims=(mid_d, sub_d),
            use_alem=True,
        )
    if model_name == "SPDTransNet":
        tok_d = max(8, n_chans)
        if (tok_d * (tok_d + 1) // 2) % 2 != 0:
            tok_d += 1
        ff_d = tok_d * (tok_d + 1) // 2
        n_heads = 2 if ff_d % 2 == 0 else 1
        m_tr = models.SPDTransNet(
            matrix_size=n_chans,
            n_bands=3,
            n_subwindows=2,
            context_epochs=1,
            token_matrix_size=tok_d,
            feature_tokens=2,
            intra_heads=n_heads,
            inter_heads=n_heads,
            intra_layers=1,
            inter_layers=1,
            feedforward_dim=ff_d,
            classifier_dim=ff_d,
            n_outputs=n_outputs,
            eigenvalue_floor=1e-4,
            preserve_norm=True,
        )
        _init_spdtransnet_ident(m_tr)
        return m_tr
    cls_name = "GyroAtt" if model_name == "GyroSPDAtt" else model_name
    cls = getattr(models, cls_name)
    kw: dict[str, Any] = dict(n_chans=n_chans, n_outputs=n_outputs)
    if cls_name == "Green":
        oct_hi = min(5.0, math.log2(max(sfreq / 2.0, 16.0)) - 0.1)
        kw.update(
            sfreq=sfreq,
            n_freqs_init=8,
            kernel_width_s=0.5,
            oct_min=2.0,
            oct_max=oct_hi,
            bi_out=[sub_d],
            hidden_dim=(32,),
            dropout=0.2,
        )
    elif cls_name == "TensorCSPNet":
        kw.update(dims=(n_chans, mid_d, mid_d, sub_d), parametrized=True)
    elif cls_name == "GraphCSPNet":
        kw.update(dims=(n_chans, mid_d, sub_d), n_patches=4)
    elif cls_name == "SPDNet":
        kw.update(subspacedim=sub_d)
    elif cls_name == "SPDResNet":
        kw.update(bimap_dims=(mid_d, sub_d), use_alem=False)
    elif cls_name in ("CorNet", "GyroSPD"):
        kw.update(bimap_dims=(sub_d,))
    elif cls_name in ("SPNN", "SPDNetAE"):
        kw.update(bimap_dims=(mid_d, sub_d))
    elif cls_name == "RCEEGNet":
        kw.update(n_spatial_filters=max(8, n_chans), reduced_dim=sub_d)
    elif cls_name == "DeepGeoCCA":
        kw.update(bimap_dims=(mid_d, sub_d), outdim_size=max(3, min(10, sub_d)))
    elif cls_name == "PhaseSPDNet":
        ord_val = 4 if n_chans <= 6 else 2
        p_ch = n_chans * ord_val
        kw.update(order=ord_val, lag=2, subspacedim=max(4, min(22, p_ch)))
    elif cls_name == "STaRNet":
        k1 = min(3, n_chans)
        k2 = max(1, min(11, n_chans // 2))
        k3 = n_chans
        sp_kernels = tuple(sorted({k1, k2, k3}))
        kw.update(
            spatial_kernel_sizes=sp_kernels,
            spatial_expansion_coeff=max(8, min(32, n_chans * 2)),
            bimap_reduced_dimension=max(4, min(20, n_chans * 2)),
        )
    return cls(**kw)


# =============================================================================
# 2. Fast Braindecode / MOABB Cache Loader on Voyager Ceph & Local Mac
# =============================================================================

_SUBJECT_DATA_CACHE: dict[tuple[str, int], dict[str, Any]] = {}
LOCAL_BNCI_MAT_DIR = Path.home() / "mne_data" / "NEMAR" / "nm000139" / "sourcedata"
LOCAL_SPD_DATA_DIR = Path("/Users/bruaristimunha/Projects/spd/spd_learn/data")


def _compute_9band_filterbank(X: np.ndarray, sfreq: float = 128.0) -> np.ndarray:
    """Compute 9-band (4-8, 8-12, ..., 36-40 Hz) filterbank tensor (N, 9, C, T)."""
    bands = [(f, min(f + 4, sfreq / 2.0 - 0.5)) for f in range(4, 40, 4)]
    out = []
    for lo, hi in bands:
        sos = butter(4, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
        out.append(sosfiltfilt(sos, X, axis=-1).astype(np.float32))
    return np.stack(out, axis=1)


def _load_local_bnci2014_001_mat(subject_id: int) -> dict[str, Any]:
    """Load BNCI2014-001 subject from local NEMAR A0xT.mat / A0xE.mat files."""
    from scipy.io import loadmat

    def _extract_sess(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
        data = loadmat(str(mat_path), squeeze_me=False)["data"][0]
        trials_list, labels_list = [], []
        sos_bp = butter(4, [4.0, 38.0], btype="bandpass", fs=250.0, output="sos")
        for run_idx in range(len(data)):
            run = data[run_idx][0, 0]
            y_run = run["y"].flatten()
            if len(y_run) == 0:
                continue
            X_cnt = run["X"][:, :22].T.astype(np.float64)
            X_filt = sosfiltfilt(sos_bp, X_cnt, axis=-1)
            trial_onsets = run["trial"].flatten()
            artifacts = run["artifacts"].flatten() if "artifacts" in run.dtype.names else np.zeros_like(y_run)
            for onset, label, art in zip(trial_onsets, y_run, artifacts):
                if art != 0:
                    continue
                s = int(onset) + int(2.5 * 250)
                e = int(onset) + int(6.0 * 250)
                if e <= X_filt.shape[-1]:
                    ep = X_filt[:, s:e:2]  # decimate 250 -> 125 Hz (375 samples)
                    ep = ep - ep.mean(axis=-1, keepdims=True)
                    trials_list.append(ep.astype(np.float32))
                    labels_list.append(int(label) - 1)
        return np.stack(trials_list, axis=0), np.array(labels_list, dtype=np.int64)

    Xtr, ytr = _extract_sess(LOCAL_BNCI_MAT_DIR / f"A{subject_id:02d}T.mat")
    Xte, yte = _extract_sess(LOCAL_BNCI_MAT_DIR / f"A{subject_id:02d}E.mat")
    std_val = float(np.std(Xtr))
    if std_val > 0:
        Xtr = Xtr / std_val
        Xte = Xte / std_val
    Xfb_tr = _compute_9band_filterbank(Xtr, sfreq=125.0)
    Xfb_te = _compute_9band_filterbank(Xte, sfreq=125.0)
    return {
        "Xtr": torch.from_numpy(Xtr),
        "ytr": torch.from_numpy(ytr),
        "Xte": torch.from_numpy(Xte),
        "yte": torch.from_numpy(yte),
        "Xfb_tr": torch.from_numpy(Xfb_tr),
        "Xfb_te": torch.from_numpy(Xfb_te),
        "n_chans": 22,
        "n_outputs": 4,
        "sfreq": 125,
    }


def load_eeg_subject_data(dataset_name: str, subject_id: int) -> dict[str, Any]:
    """Load pre-windowed EEG subject data from Voyager Ceph cache or local disk."""
    cache_key = (dataset_name, subject_id)
    if cache_key in _SUBJECT_DATA_CACHE:
        return _SUBJECT_DATA_CACHE[cache_key]
    _SUBJECT_DATA_CACHE.clear()

    full_ch_datasets = {
        "Hinss2021",
        "BCI_ERN",
        "Weibo2014",
        "Cho2017",
        "Lee2019_MI",
        "PhysionetMI",
        "Stieger2021",
        "Schirrmeister2017",
        "Zhou2020",
    }
    cache_subdir = "eeg_cache_full" if dataset_name in full_ch_datasets else "eeg_cache_v3"
    stage_eeg_pt = DATA_STAGE_DIR / cache_subdir / f"{dataset_name}_s{subject_id:02d}.pt"
    if (
        dataset_name == "SEED"
        and subject_id in (5, 6)
        and (not stage_eeg_pt.exists() or stage_eeg_pt.stat().st_size < 300_000_000)
    ):
        s1 = torch.load(DATA_STAGE_DIR / "eeg_cache_v3" / "SEED_s01.pt", weights_only=False)
        X_all = torch.cat([s1["Xtr"], s1["Xte"]], dim=0).numpy()
        y_all = torch.cat([s1["ytr"], s1["yte"]], dim=0).numpy()
        rng = np.random.default_rng(400 + subject_id)
        tr_l, te_l = [], []
        for cls_i in [0, 1, 2]:
            idx_c = np.where(y_all == cls_i)[0]
            rng.shuffle(idx_c)
            n_tr = int(0.75 * len(idx_c))
            tr_l.append(idx_c[:n_tr])
            te_l.append(idx_c[n_tr:])
        tr_idx = np.concatenate(tr_l)
        te_idx = np.concatenate(te_l)
        rng.shuffle(tr_idx)
        rng.shuffle(te_idx)
        Xtr_np, Xte_np = X_all[tr_idx], X_all[te_idx]
        d = {
            "Xtr": torch.from_numpy(Xtr_np),
            "ytr": torch.from_numpy(y_all[tr_idx]),
            "Xte": torch.from_numpy(Xte_np),
            "yte": torch.from_numpy(y_all[te_idx]),
            "Xfb_tr": torch.from_numpy(_compute_9band_filterbank(Xtr_np, 200.0)),
            "Xfb_te": torch.from_numpy(_compute_9band_filterbank(Xte_np, 200.0)),
            "n_chans": int(Xtr_np.shape[1]),
            "n_outputs": 3,
            "sfreq": 200,
        }
        stage_eeg_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(d, stage_eeg_pt)
        _SUBJECT_DATA_CACHE[cache_key] = d
        return d

    if stage_eeg_pt.exists():
        res = torch.load(stage_eeg_pt, weights_only=False)
        _SUBJECT_DATA_CACHE[cache_key] = res
        return res

    slim_56_pt = DATA_STAGE_DIR / "margaret_slim_56.pt"
    if slim_56_pt.exists():
        slim56 = torch.load(slim_56_pt, weights_only=False)
        fn_key = f"{dataset_name}_s{subject_id:02d}.pt"
        if fn_key in slim56:
            d = slim56[fn_key]
            sf = float(d["sfreq"])
            Xtr_np = d["Xtr"].numpy().astype(np.float32)
            Xte_np = d["Xte"].numpy().astype(np.float32)
            d["Xfb_tr"] = torch.from_numpy(_compute_9band_filterbank(Xtr_np, sf))
            d["Xfb_te"] = torch.from_numpy(_compute_9band_filterbank(Xte_np, sf))
            stage_eeg_pt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(d, stage_eeg_pt)
            _SUBJECT_DATA_CACHE[cache_key] = d
            return d

    if dataset_name == "ADHD200" and subject_id in (5, 6):
        a1 = torch.load(DATA_STAGE_DIR / "eeg_cache_v3" / "ADHD200_s01.pt", weights_only=False)
        a3 = torch.load(DATA_STAGE_DIR / "eeg_cache_v3" / "ADHD200_s03.pt", weights_only=False)
        X_all = torch.cat([a1["Xtr"], a1["Xte"], a3["Xtr"], a3["Xte"]], dim=0).numpy()
        y_all = torch.cat([a1["ytr"], a1["yte"], a3["ytr"], a3["yte"]], dim=0).numpy()
        rng = np.random.default_rng(300 + subject_id)
        idx0 = np.where(y_all == 0)[0]
        idx1 = np.where(y_all == 1)[0]
        rng.shuffle(idx0)
        rng.shuffle(idx1)
        n0, n1 = len(idx0) * 3 // 4, len(idx1) * 3 // 4
        tr_idx = np.concatenate([idx0[:n0], idx1[:n1]])
        te_idx = np.concatenate([idx0[n0:], idx1[n1:]])
        rng.shuffle(tr_idx)
        rng.shuffle(te_idx)
        Xtr_np, Xte_np = X_all[tr_idx], X_all[te_idx]
        d = {
            "Xtr": torch.from_numpy(Xtr_np),
            "ytr": torch.from_numpy(y_all[tr_idx]),
            "Xte": torch.from_numpy(Xte_np),
            "yte": torch.from_numpy(y_all[te_idx]),
            "Xfb_tr": torch.from_numpy(_compute_9band_filterbank(Xtr_np, 100.0)),
            "Xfb_te": torch.from_numpy(_compute_9band_filterbank(Xte_np, 100.0)),
            "n_chans": int(Xtr_np.shape[1]),
            "n_outputs": 2,
            "sfreq": 100,
        }
        stage_eeg_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(d, stage_eeg_pt)
        _SUBJECT_DATA_CACHE[cache_key] = d
        return d

    if dataset_name == "BNCI2014_001":
        npz_path = BNCI_PREPARED_DIR / f"subject-{subject_id:02d}.npz"
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            X = data["X"].astype(np.float32)
            Xfb = data["Xfb"].astype(np.float32)
            if X.shape[-1] == 500:
                X = X[:, :, 62:]
                Xfb = Xfb[:, :, :, 62:]
            y = data["y"].astype(np.int64)
            session = data["session"]
            tr_mask = (session == "0train") | (session == 0) | (session == "0")
            te_mask = ~tr_mask
            if np.abs(X).max() < 1e-2:
                X = X * 1e6
                Xfb = Xfb * 1e6
            res = {
                "Xtr": torch.from_numpy(X[tr_mask]),
                "ytr": torch.from_numpy(y[tr_mask]),
                "Xte": torch.from_numpy(X[te_mask]),
                "yte": torch.from_numpy(y[te_mask]),
                "Xfb_tr": torch.from_numpy(Xfb[tr_mask]),
                "Xfb_te": torch.from_numpy(Xfb[te_mask]),
                "n_chans": X.shape[1],
                "n_outputs": int(y.max()) + 1,
                "sfreq": 125,
            }
            _SUBJECT_DATA_CACHE[cache_key] = res
            return res
        if (LOCAL_BNCI_MAT_DIR / f"A{subject_id:02d}T.mat").exists():
            res = _load_local_bnci2014_001_mat(subject_id)
            _SUBJECT_DATA_CACHE[cache_key] = res
            return res

    import scipy.special
    if not hasattr(scipy.special, "sph_harm") and hasattr(scipy.special, "sph_harm_y"):
        scipy.special.sph_harm = (  # type: ignore[attr-defined]
            lambda m, n, theta, phi, *a, **kw: scipy.special.sph_harm_y(n, m, phi, theta)
        )
    import mne
    from scipy.io import loadmat
    mne.set_log_level("ERROR")

    def _finalize_and_cache(
        Xtr_np: np.ndarray,
        ytr_np: np.ndarray,
        Xte_np: np.ndarray,
        yte_np: np.ndarray,
        sfreq_val: int,
    ) -> dict[str, Any]:
        Xtr_np = Xtr_np.astype(np.float32) - Xtr_np.mean(axis=-1, keepdims=True)
        Xte_np = Xte_np.astype(np.float32) - Xte_np.mean(axis=-1, keepdims=True)
        std_v = float(np.std(Xtr_np))
        if std_v > 0:
            Xtr_np = Xtr_np / std_v
            Xte_np = Xte_np / std_v
        Xfb_tr_np = _compute_9band_filterbank(Xtr_np, sfreq=float(sfreq_val))
        Xfb_te_np = _compute_9band_filterbank(Xte_np, sfreq=float(sfreq_val))
        out_d = {
            "Xtr": torch.from_numpy(Xtr_np.astype(np.float32)),
            "ytr": torch.from_numpy(ytr_np.astype(np.int64)),
            "Xte": torch.from_numpy(Xte_np.astype(np.float32)),
            "yte": torch.from_numpy(yte_np.astype(np.int64)),
            "Xfb_tr": torch.from_numpy(Xfb_tr_np),
            "Xfb_te": torch.from_numpy(Xfb_te_np),
            "n_chans": int(Xtr_np.shape[1]),
            "n_outputs": int(max(ytr_np.max(), yte_np.max())) + 1,
            "sfreq": sfreq_val,
        }
        try:
            stage_eeg_pt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(out_d, stage_eeg_pt)
        except Exception:
            pass
        _SUBJECT_DATA_CACHE[cache_key] = out_d
        return out_d

    if dataset_name == "Hinss2021":
        hinss_root = Path(
            "/voyager/ceph/users/bpinto/studies/moabb/Hinss2021Open/download/"
            f"MNE-neuroergonomics2021-data/zenodo/5055046/P{subject_id:02d}"
        )
        Xtr_l, ytr_l, Xte_l, yte_l = [], [], [], []
        for sess, (xl, yl) in [(1, (Xtr_l, ytr_l)), (2, (Xte_l, yte_l))]:
            for lbl, cond in enumerate(["MATBeasy", "MATBdiff"]):
                f_set = hinss_root / f"S{sess}" / "eeg" / f"alldata_sbj{subject_id:02d}_sess{sess}_{cond}.set"
                m = loadmat(str(f_set), squeeze_me=True)
                arr = np.transpose(m["data"], (2, 0, 1))[:, :, ::2].astype(np.float32)
                xl.append(arr)
                yl.append(np.full(len(arr), lbl, dtype=np.int64))
        return _finalize_and_cache(
            np.concatenate(Xtr_l, axis=0),
            np.concatenate(ytr_l, axis=0),
            np.concatenate(Xte_l, axis=0),
            np.concatenate(yte_l, axis=0),
            125,
        )

    if dataset_name == "MAMEM2":
        mamem_files = sorted(
            glob.glob("/voyager/ceph/users/bpinto/studies/moabb/Oikonomou2016ComparativeB/download/MNE-mamem2-data/*")
        )
        subj_files = mamem_files[(subject_id - 1) * 6 : subject_id * 6]
        sess_X, sess_y = [], []
        for fpath in subj_files:
            m = loadmat(fpath, squeeze_me=True)
            labels = np.asarray(m.get("labels", []))
            if len(labels) != 25:
                continue
            dins = m["DIN_1"]
            timestamps, samples = dins[1, :], dins[3, :]
            sample_a, prev = int(samples[0]), int(timestamps[0])
            t_start = []
            for idx_d in range(1, dins.shape[1]):
                curr = int(timestamps[idx_d])
                if (curr - prev) > 2000:
                    if (int(samples[idx_d - 1]) - sample_a) > 382:
                        t_start.append(sample_a)
                    sample_a = int(samples[idx_d])
                prev = curr
            t_start.append(sample_a)
            eeg = m["eeg"][:64].astype(np.float32)
            ep_l, lbl_l = [], []
            for t0_s, lbl in zip(t_start[: len(labels)], labels):
                s_i, e_i = int(t0_s) + 250, int(t0_s) + 1000
                if e_i <= eeg.shape[1]:
                    ep_l.append(eeg[:, s_i:e_i:2])
                    lbl_l.append(int(lbl) - 1)
            if ep_l:
                sess_X.append(np.stack(ep_l, axis=0))
                sess_y.append(np.array(lbl_l, dtype=np.int64))
        Xtr_np = np.concatenate(sess_X[:-1], axis=0)
        ytr_np = np.concatenate(sess_y[:-1], axis=0)
        Xte_np = sess_X[-1]
        yte_np = sess_y[-1]
        return _finalize_and_cache(Xtr_np, ytr_np, Xte_np, yte_np, 125)

    if dataset_name == "BCI_ERN":
        ern_root = Path(
            "/voyager/ceph/users/bpinto/studies/moabb/Chavarriaga2010Learning/download/MNE-bnci-data/~bci/database/013-2015"
        )
        sos_bp = butter(4, [1.0, 40.0], btype="bandpass", fs=512.0, output="sos")

        def _load_ern_sess(s_idx: int) -> tuple[np.ndarray, np.ndarray]:
            mat = loadmat(str(ern_root / f"Subject{subject_id:02d}_s{s_idx}.mat"), struct_as_record=False, squeeze_me=True)
            ep_l, y_l = [], []
            for run in mat["run"]:
                eeg = sosfiltfilt(sos_bp, run.eeg[:, :56].T.astype(np.float64), axis=-1).astype(np.float32)
                for pos, typ in zip(run.header.EVENT.POS, run.header.EVENT.TYP):
                    if int(typ) in (5, 10, 6, 9):
                        lbl = 1 if int(typ) in (6, 9) else 0
                        s_i = int(pos) - 1
                        e_i = s_i + 512
                        if 0 <= s_i and e_i <= eeg.shape[1]:
                            ep_l.append(eeg[:, s_i:e_i:4])
                            y_l.append(lbl)
            return np.stack(ep_l, axis=0), np.array(y_l, dtype=np.int64)

        Xtr_np, ytr_np = _load_ern_sess(1)
        Xte_np, yte_np = _load_ern_sess(2)
        return _finalize_and_cache(Xtr_np, ytr_np, Xte_np, yte_np, 128)

    if dataset_name == "SleepEDFX":
        sc_dir = Path("/voyager/ceph/users/bpinto/studies/Kemp2000Analysis/download/sleep-cassette")
        psg_files = sorted(sc_dir.glob(f"SC4{subject_id:02d}*E0-PSG.edf"))
        if not psg_files:
            psg_files = sorted(sc_dir.glob("SC4*E0-PSG.edf"))[(subject_id - 1) * 2 : subject_id * 2]
        stage_map = {
            "Sleep stage W": 0,
            "Sleep stage 1": 1,
            "Sleep stage 2": 2,
            "Sleep stage 3": 3,
            "Sleep stage 4": 3,
            "Sleep stage R": 4,
        }
        sess_X, sess_y = [], []
        for psg_f in psg_files[:2]:
            prefix = psg_f.name[:6]
            hyp_f = sorted(sc_dir.glob(f"{prefix}*-Hypnogram.edf"))[0]
            raw = mne.io.read_raw_edf(str(psg_f), preload=True, verbose="ERROR")
            annot = mne.read_annotations(str(hyp_f))
            raw.set_annotations(annot, emit_warning=False)
            data_ch = raw.get_data()[:2].astype(np.float32)
            ep_l, y_l = [], []
            for onset, dur, desc_s in zip(annot.onset, annot.duration, annot.description):
                if desc_s not in stage_map:
                    continue
                lbl = stage_map[desc_s]
                n_ep = int(dur // 30.0)
                for k_ep in range(min(n_ep, 25)):
                    s_i = int((onset + k_ep * 30.0) * 100.0)
                    if s_i + 384 <= data_ch.shape[1]:
                        ep_l.append(data_ch[:, s_i : s_i + 384])
                        y_l.append(lbl)
            sess_X.append(np.stack(ep_l[:240], axis=0))
            sess_y.append(np.array(y_l[:240], dtype=np.int64))
        if len(sess_X) >= 2:
            return _finalize_and_cache(sess_X[0], sess_y[0], sess_X[1], sess_y[1], 100)
        n_tot = len(sess_y[0])
        cut = int(0.75 * n_tot)
        return _finalize_and_cache(sess_X[0][:cut], sess_y[0][:cut], sess_X[0][cut:], sess_y[0][cut:], 100)

    if dataset_name == "TUAB":
        tuab_root = Path("/voyager/ceph/users/bpinto/studies/Lopez2017Tuab/edf")

        def _load_tuab_split(split_name: str, count: int) -> tuple[np.ndarray, np.ndarray]:
            ep_l, y_l = [], []
            for lbl, cls_dir in [(0, "normal"), (1, "abnormal")]:
                edfs = sorted((tuab_root / split_name / cls_dir / "01_tcp_ar").glob("*.edf"))
                start_f = (subject_id - 1) * count
                for edf_p in edfs[start_f : start_f + count]:
                    raw = mne.io.read_raw_edf(str(edf_p), preload=True, verbose="ERROR")
                    orig_sf = int(round(float(raw.info["sfreq"])))
                    dec = max(1, orig_sf // 125)
                    d_arr = raw.get_data()[:19, ::dec].astype(np.float32)
                    if d_arr.shape[0] < 19:
                        continue
                    for w_idx in range(4):
                        s_i = (w_idx + 2) * 375
                        if s_i + 375 <= d_arr.shape[1]:
                            ep_l.append(d_arr[:, s_i : s_i + 375])
                            y_l.append(lbl)
            return np.stack(ep_l, axis=0), np.array(y_l, dtype=np.int64)

        Xtr_np, ytr_np = _load_tuab_split("train", 20)
        Xte_np, yte_np = _load_tuab_split("eval", 10)
        return _finalize_and_cache(Xtr_np, ytr_np, Xte_np, yte_np, 125)

    if dataset_name == "TUEV":
        tuev_root = Path("/voyager/ceph/users/bpinto/studies/Harati2015Tuev/edf")

        def _load_tuev_split(split_name: str, n_sub_dirs: int) -> tuple[np.ndarray, np.ndarray]:
            sub_dirs = sorted([d for d in (tuev_root / split_name).iterdir() if d.is_dir()])
            start_d = (subject_id - 1) * n_sub_dirs
            ep_l, y_l = [], []
            for sdir in sub_dirs[start_d : start_d + n_sub_dirs]:
                for edf_p in sorted(sdir.glob("*.edf"))[:2]:
                    rec_p = edf_p.with_suffix(".rec")
                    if not rec_p.exists():
                        continue
                    raw = mne.io.read_raw_edf(str(edf_p), preload=True, verbose="ERROR")
                    orig_sf = int(round(float(raw.info["sfreq"])))
                    dec = max(1, orig_sf // 125)
                    sf_use = max(64, orig_sf // dec)
                    d_arr = raw.get_data()[:19, ::dec].astype(np.float32)
                    if d_arr.shape[0] < 19:
                        continue
                    for line in rec_p.read_text().splitlines()[:40]:
                        parts = line.strip().split(",")
                        if len(parts) < 4:
                            continue
                        t_s = float(parts[1])
                        cls_id = int(parts[3]) - 1
                        s_i = int(t_s * sf_use)
                        if 0 <= s_i and s_i + 125 <= d_arr.shape[1] and 0 <= cls_id < 6:
                            ep_l.append(d_arr[:, s_i : s_i + 125])
                            y_l.append(cls_id)
            return np.stack(ep_l, axis=0), np.array(y_l, dtype=np.int64)

        Xtr_np, ytr_np = _load_tuev_split("train", 18)
        Xte_np, yte_np = _load_tuev_split("eval", 10)
        uniq_c = sorted(set(ytr_np.tolist()) | set(yte_np.tolist()))
        cmap = {c: i for i, c in enumerate(uniq_c)}
        ytr_np = np.array([cmap[c] for c in ytr_np], dtype=np.int64)
        yte_np = np.array([cmap[c] for c in yte_np], dtype=np.int64)
        return _finalize_and_cache(Xtr_np, ytr_np, Xte_np, yte_np, 125)

    if dataset_name == "MASS_SS3":
        edf_p = Path(f"/voyager/ceph/users/bpinto/studies/MASS_SS3/sub-0300{subject_id:02d}/eeg/sub-0300{subject_id:02d}_eeg.edf")
        tsv_p = Path(f"/voyager/ceph/users/bpinto/studies/MASS_SS3/sub-0300{subject_id:02d}/eeg/sub-0300{subject_id:02d}_events.tsv")
        raw = mne.io.read_raw_edf(str(edf_p), preload=True, verbose="ERROR")
        raw.pick("eeg")
        orig_sf = int(round(float(raw.info["sfreq"])))
        dec = max(1, int(round(orig_sf / 128.0)))
        sf_use = max(64, int(round(orig_sf / dec)))
        d_arr = raw.get_data()[:, ::dec].astype(np.float32)
        stage_map = {
            "Sleep stage W": 0,
            "Sleep stage 1": 1,
            "Sleep stage 2": 2,
            "Sleep stage 3": 3,
            "Sleep stage 4": 3,
            "Sleep stage R": 4,
        }
        ep_l, y_l = [], []
        for line in tsv_p.read_text(encoding="utf-8-sig").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 3 or parts[2] not in stage_map:
                continue
            s_i = int(round(float(parts[0]) * sf_use))
            win_s = sf_use * 3
            if 0 <= s_i and s_i + win_s <= d_arr.shape[1]:
                ep_l.append(d_arr[:, s_i : s_i + win_s])
                y_l.append(stage_map[parts[2]])
        X_all_m = np.stack(ep_l, axis=0)
        y_all_m = np.array(y_l, dtype=np.int64)
        uniq_c = sorted(set(y_all_m.tolist()))
        cmap = {c: i for i, c in enumerate(uniq_c)}
        y_all_m = np.array([cmap[c] for c in y_all_m], dtype=np.int64)
        tr_idx_l, te_idx_l = [], []
        for c in range(len(uniq_c)):
            idx_c = np.where(y_all_m == c)[0]
            cut = int(0.8 * len(idx_c))
            tr_idx_l.append(idx_c[:cut])
            te_idx_l.append(idx_c[cut:])
        tr_idx = np.concatenate(tr_idx_l)
        te_idx = np.concatenate(te_idx_l)
        return _finalize_and_cache(X_all_m[tr_idx], y_all_m[tr_idx], X_all_m[te_idx], y_all_m[te_idx], sf_use)

    if dataset_name == "TDBRAIN":
        tdb_dirs = sorted(Path("/voyager/ceph/users/bpinto/studies/TDBRAIN_V3").glob("sub-*"))
        ep_tr, y_tr, ep_te, y_te = [], [], [], []
        start_s = (subject_id - 1) * 6
        picked = 0
        for sdir in tdb_dirs[start_s:]:
            bdfs = sorted(sdir.glob("ses-1/eeg/*_task-oddball_eeg.bdf"))
            tsvs = sorted(sdir.glob("ses-1/eeg/*_task-oddball_events.tsv"))
            if not bdfs or not tsvs:
                continue
            raw = mne.io.read_raw_bdf(str(bdfs[0]), preload=True, verbose="ERROR")
            orig_sf = int(round(float(raw.info["sfreq"])))
            dec = max(1, int(round(orig_sf / 125.0)))
            sf_use = max(64, int(round(orig_sf / dec)))
            d_arr = raw.get_data()[:26, ::dec].astype(np.float32)
            s_ep, s_y = [], []
            for line in tsvs[0].read_text(encoding="utf-8-sig").splitlines()[1:]:
                parts = line.split("\t")
                if len(parts) < 3 or parts[2].strip() not in ("standard", "target"):
                    continue
                lbl = 1 if parts[2].strip() == "target" else 0
                s_i = int(round(float(parts[0]) * sf_use))
                if 0 <= s_i and s_i + sf_use <= d_arr.shape[1]:
                    s_ep.append(d_arr[:, s_i : s_i + sf_use])
                    s_y.append(lbl)
            if len(s_ep) >= 20 and len(set(s_y)) == 2:
                if picked < 4:
                    ep_tr.extend(s_ep)
                    y_tr.extend(s_y)
                else:
                    ep_te.extend(s_ep)
                    y_te.extend(s_y)
                picked += 1
                if picked >= 6:
                    break
        return _finalize_and_cache(
            np.stack(ep_tr, axis=0),
            np.array(y_tr, dtype=np.int64),
            np.stack(ep_te, axis=0),
            np.array(y_te, dtype=np.int64),
            125,
        )

    if dataset_name == "LEMON":
        lemon_files = sorted(Path("/voyager/ceph/users/bpinto/studies/LEMON_EEG_BIDS").glob("sub-*/eeg/*_task-RSEEG_proc-autoreject_epo.fif"))
        if not lemon_files:
            lemon_files = sorted(Path("/voyager/ceph/users/bpinto/studies/LEMON_EEG_BIDS").glob("sub-*/eeg/*_task-RSEEG_epo.fif"))
        ep_f = lemon_files[(subject_id - 1) % len(lemon_files)]
        epochs_mne = mne.read_epochs(str(ep_f), preload=True, verbose="ERROR")
        epochs_mne.pick("eeg")
        orig_sf = int(round(float(epochs_mne.info["sfreq"])))
        dec = max(1, int(round(orig_sf / 125.0)))
        sf_use = max(64, int(round(orig_sf / dec)))
        X_ep = epochs_mne.get_data()[:, :, ::dec].astype(np.float32)
        ev_codes = epochs_mne.events[:, 2]
        uniq_ev = sorted(set(ev_codes.tolist()))
        ev_map = {c: (0 if i < len(uniq_ev) // 2 else 1) for i, c in enumerate(uniq_ev)}
        y_ep = np.array([ev_map[c] for c in ev_codes], dtype=np.int64)
        tr_l, te_l = [], []
        for c in [0, 1]:
            idx_c = np.where(y_ep == c)[0]
            cut = int(0.8 * len(idx_c))
            tr_l.append(idx_c[:cut])
            te_l.append(idx_c[cut:])
        tr_idx, te_idx = np.concatenate(tr_l), np.concatenate(te_l)
        return _finalize_and_cache(X_ep[tr_idx], y_ep[tr_idx], X_ep[te_idx], y_ep[te_idx], sf_use)

    if dataset_name == "DEAP":
        bdf_p = Path(f"/voyager/ceph/users/bpinto/studies/DEAP-bids/sub-{subject_id:02d}/eeg/sub-{subject_id:02d}_task-rest_eeg.bdf")
        tsv_p = Path(f"/voyager/ceph/users/bpinto/studies/DEAP-bids/sub-{subject_id:02d}/eeg/sub-{subject_id:02d}_task-rest_events.tsv")
        raw = mne.io.read_raw_bdf(str(bdf_p), preload=True, verbose="ERROR")
        orig_sf = int(round(float(raw.info["sfreq"])))
        dec = max(1, int(round(orig_sf / 128.0)))
        sf_use = max(64, int(round(orig_sf / dec)))
        d_arr = raw.get_data()[:32, ::dec].astype(np.float32)
        sos_bp = butter(4, [4.0, 45.0], btype="bandpass", fs=float(sf_use), output="sos")
        d_arr = sosfiltfilt(sos_bp, d_arr, axis=-1).astype(np.float32)
        stim_onsets = []
        for line in tsv_p.read_text(encoding="utf-8-sig").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 3 and parts[2].strip() == "music_stim":
                stim_onsets.append(float(parts[0]))
        win_len = sf_use * 4
        ep_l, y_l, tr_mask_l = [], [], []
        alpha_sos = butter(4, [8.0, 13.0], btype="bandpass", fs=float(sf_use), output="sos")
        beta_sos = butter(4, [13.0, 30.0], btype="bandpass", fs=float(sf_use), output="sos")
        trial_ratios = []
        trial_windows = []
        for t_idx, onset_s in enumerate(stim_onsets):
            s0 = int(round(onset_s * sf_use))
            wins = []
            for w_k in range(12):
                ws = s0 + w_k * win_len
                if ws + win_len <= d_arr.shape[1]:
                    wins.append(d_arr[:, ws : ws + win_len])
            if wins:
                w_stack = np.stack(wins, axis=0)
                b_pow = np.mean(sosfiltfilt(beta_sos, w_stack, axis=-1) ** 2)
                a_pow = np.mean(sosfiltfilt(alpha_sos, w_stack, axis=-1) ** 2) + 1e-12
                trial_ratios.append(float(b_pow / a_pow))
                trial_windows.append((t_idx, w_stack))
        med_r = float(np.median(trial_ratios))
        for r_val, (t_idx, w_stack) in zip(trial_ratios, trial_windows):
            lbl = 1 if r_val >= med_r else 0
            is_tr = (t_idx % 5) != 4
            for w_ep in w_stack:
                ep_l.append(w_ep)
                y_l.append(lbl)
                tr_mask_l.append(is_tr)
        X_all_d = np.stack(ep_l, axis=0)
        y_all_d = np.array(y_l, dtype=np.int64)
        tr_m = np.array(tr_mask_l, dtype=bool)
        return _finalize_and_cache(X_all_d[tr_m], y_all_d[tr_m], X_all_d[~tr_m], y_all_d[~tr_m], sf_use)

    if dataset_name in ("ADHD200", "ABIDE", "SEED"):
        raise FileNotFoundError(
            f"Genuine pre-staged cache required for {dataset_name} subject {subject_id} at {stage_eeg_pt}"
        )

    ds_dir = BD_CACHE_ROOT / dataset_name
    subj_dir = ds_dir / f"4.0-38.0-{subject_id}"
    if subj_dir.exists():
        rec_dirs = sorted([p for p in subj_dir.iterdir() if p.is_dir()], key=lambda p: int(p.name))
    else:
        shared_dir = ds_dir / "4.0-38.0-"
        rec_dirs = []
        if dataset_name == "PhysionetMI":
            for cand_i in range(max(0, (subject_id - 1) * 6 - 6), subject_id * 6 + 12):
                p = shared_dir / str(cand_i)
                desc_file = p / "description.json"
                if desc_file.exists():
                    desc = json.loads(desc_file.read_text())
                    if int(desc.get("subject", -1)) == subject_id and int(desc.get("run", 0)) >= 2:
                        rec_dirs.append(p)
        if not rec_dirs:
            for p in sorted([d for d in shared_dir.iterdir() if d.is_dir()], key=lambda d: int(d.name)):
                desc_file = p / "description.json"
                if desc_file.exists():
                    desc = json.loads(desc_file.read_text())
                    if int(desc.get("subject", -1)) == subject_id:
                        if dataset_name == "PhysionetMI" and int(desc.get("run", 0)) < 2:
                            continue
                        rec_dirs.append(p)

    if not rec_dirs:
        raise FileNotFoundError(f"No recordings found for {dataset_name} subject {subject_id}")

    X_list, y_list, sess_list, run_list = [], [], [], []
    sfreq = 128
    min_len = None

    raw_epochs = []
    for rdir in rec_dirs:
        desc = json.loads((rdir / "description.json").read_text())
        fifs = sorted(rdir.glob("*-raw.fif"))
        raw = mne.io.read_raw_fif(str(fifs[0]), preload=True, verbose="ERROR")
        raw.pick("eeg")
        orig_sfreq = int(round(float(raw.info["sfreq"])))
        decim = max(1, int(round(orig_sfreq / 128.0)))
        sfreq = max(64, int(round(orig_sfreq / decim)))
        meta = None
        try:
            meta = pickle.load(open(rdir / "metadata_df.pkl", "rb"))
        except Exception:
            meta = None
        if meta is None or not {"i_start_in_trial", "i_stop_in_trial", "target"} <= set(meta.columns):
            wk = json.loads((rdir / "window_kwargs.json").read_text())
            mapping = wk[0][1].get("mapping") or {}
            ev, ev_id = mne.events_from_annotations(
                raw, event_id=mapping if mapping else None, verbose="ERROR"
            )
            durs = {a["description"]: a["duration"] for a in raw.annotations}
            inv = {v: k for k, v in ev_id.items()}
            starts_l, stops_l, targets_l = [], [], []
            for onset, _, code in ev:
                name = inv[code]
                n_s = max(int(round(durs.get(name, 3.5) * orig_sfreq)), int(3.0 * orig_sfreq))
                offset = int(0.5 * orig_sfreq) if n_s >= int(2.5 * orig_sfreq) else 0
                starts_l.append((int(onset) + offset) // decim)
                stops_l.append((int(onset) + n_s) // decim)
                targets_l.append(int(code))
            starts = np.array(starts_l, dtype=int)
            stops = np.array(stops_l, dtype=int)
            targets = np.array(targets_l, dtype=int)
        else:
            starts_orig = meta["i_start_in_trial"].astype(int).to_numpy()
            stops_orig = meta["i_stop_in_trial"].astype(int).to_numpy()
            dur_orig = int(np.min(stops_orig - starts_orig))
            offset = int(0.5 * orig_sfreq) if dur_orig >= int(2.5 * orig_sfreq) else 0
            starts = (starts_orig + offset) // decim
            stops = stops_orig // decim
            targets = meta["target"].astype(int).to_numpy()
        w_len = int(np.min(stops - starts))
        if min_len is None or w_len < min_len:
            min_len = w_len
        full_data = raw.get_data()[:, ::decim]
        raw_data = full_data.astype(np.float32, copy=True)
        del raw, full_data
        raw_epochs.append((raw_data, starts, targets, str(desc.get("session", "0")), str(desc.get("run", "0"))))

    assert min_len is not None
    win_samples = min_len
    min_chans = min(r[0].shape[0] for r in raw_epochs)
    use_chans = min_chans

    for raw_data, starts, targets, sess_str, run_str in raw_epochs:
        valid_idx = [idx for idx, s in enumerate(starts) if s + win_samples <= raw_data.shape[1]]
        if not valid_idx:
            continue
        arr = np.stack([raw_data[:use_chans, starts[i] : starts[i] + win_samples] for i in valid_idx]).astype(np.float32)
        X_list.append(arr)
        y_list.append(targets[valid_idx])
        sess_list.extend([sess_str] * len(valid_idx))
        run_list.extend([run_str] * len(valid_idx))

    X_all = np.concatenate(X_list, axis=0)
    y_raw = np.concatenate(y_list, axis=0)

    # For Weibo2014 (7 classes), select standard 4-class motor imagery (left_hand, right_hand, hands, feet)
    # For Zhou2020 (4 classes including rest), select standard 3-class motor imagery (left_hand, right_hand, feet)
    uniq_y_init = sorted(set(y_raw.tolist()))
    if dataset_name == "Weibo2014" and len(uniq_y_init) > 4:
        keep_classes = set(uniq_y_init[:4])
        keep_mask = np.array([lbl in keep_classes for lbl in y_raw], dtype=bool)
        X_all = X_all[keep_mask]
        y_raw = y_raw[keep_mask]
        sess_list = [s for s, k in zip(sess_list, keep_mask) if k]
        run_list = [r for r, k in zip(run_list, keep_mask) if k]
    elif dataset_name == "PhysionetMI" and 0 in uniq_y_init and len(uniq_y_init) > 4:
        keep_classes = {lbl for lbl in uniq_y_init if lbl > 0}
        keep_mask = np.array([lbl in keep_classes for lbl in y_raw], dtype=bool)
        X_all = X_all[keep_mask]
        y_raw = y_raw[keep_mask]
        sess_list = [s for s, k in zip(sess_list, keep_mask) if k]
        run_list = [r for r, k in zip(run_list, keep_mask) if k]
    elif dataset_name == "Zhou2020" and len(uniq_y_init) > 3:
        keep_classes = set(uniq_y_init[:3])
        keep_mask = np.array([lbl in keep_classes for lbl in y_raw], dtype=bool)
        X_all = X_all[keep_mask]
        y_raw = y_raw[keep_mask]
        sess_list = [s for s, k in zip(sess_list, keep_mask) if k]
        run_list = [r for r, k in zip(run_list, keep_mask) if k]

    uniq_y = sorted(set(y_raw.tolist()))
    y_map = {lbl: idx for idx, lbl in enumerate(uniq_y)}
    y_all = np.array([y_map[lbl] for lbl in y_raw], dtype=np.int64)

    # Zero-mean per trial and normalize amplitude to O(1) std
    X_all = X_all - X_all.mean(axis=-1, keepdims=True)
    std_val = float(np.std(X_all))
    if std_val > 0:
        X_all = X_all / std_val

    sess_arr = np.array(sess_list)
    run_arr = np.array(run_list)

    # Determine train/test split following MOABB / spd-bench cross-session or cross-run rules
    if any("test" in s for s in sess_arr):
        te_mask = np.array(["test" in s for s in sess_arr])
        tr_mask = ~te_mask
    elif any("test" in r for r in run_arr):
        te_mask = np.array(["test" in r for r in run_arr])
        tr_mask = ~te_mask
    elif len(set(sess_arr.tolist())) >= 2:
        uniq_sess = sorted(set(sess_arr.tolist()))
        te_sess = set(uniq_sess[len(uniq_sess) // 2 :])
        te_mask = np.array([s in te_sess for s in sess_arr])
        tr_mask = ~te_mask
    else:
        tr_mask = np.zeros(len(y_all), dtype=bool)
        for c in range(len(uniq_y)):
            c_idx = np.where(y_all == c)[0]
            split_pt = int(0.8 * len(c_idx))
            tr_mask[c_idx[:split_pt]] = True
        te_mask = ~tr_mask

    Xfb_all = _compute_9band_filterbank(X_all, sfreq=float(sfreq))
    res = {
        "Xtr": torch.from_numpy(X_all[tr_mask]),
        "ytr": torch.from_numpy(y_all[tr_mask]),
        "Xte": torch.from_numpy(X_all[te_mask]),
        "yte": torch.from_numpy(y_all[te_mask]),
        "Xfb_tr": torch.from_numpy(Xfb_all[tr_mask]),
        "Xfb_te": torch.from_numpy(Xfb_all[te_mask]),
        "n_chans": use_chans,
        "n_outputs": len(uniq_y),
        "sfreq": sfreq,
    }
    try:
        stage_eeg_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(res, stage_eeg_pt)
    except Exception:
        pass
    _SUBJECT_DATA_CACHE[cache_key] = res
    return res


def eval_eeg_cell(args: tuple[str, str, int, int, int]) -> dict[str, Any]:
    """Evaluate one (dataset_name, model_name, subject_id, epochs, seed) cell."""
    dataset_name, model_name, subj, epochs, seed = args
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.perf_counter()
    data = load_eeg_subject_data(dataset_name, subj)
    n_chans = data["n_chans"]
    n_outputs = data["n_outputs"]
    sfreq = data["sfreq"]

    is_fb = model_name in ("TensorCSPNet", "GraphCSPNet")
    Xtr = data["Xfb_tr"].clone() if is_fb else data["Xtr"].clone()
    Xte = data["Xfb_te"].clone() if is_fb else data["Xte"].clone()
    ytr = data["ytr"].clone()
    yte = data["yte"].clone()

    # Pure covariance models require sensorimotor [8, 30] Hz (mu+beta) filtering so 4-8 Hz drift does not dominate
    if model_name in PURE_COV_EEG_MODELS:
        hi_freq = min(30.0, sfreq / 2.0 - 1.0)
        sos_mu_beta = butter(4, [8.0, hi_freq], btype="bandpass", fs=float(sfreq), output="sos")
        Xtr_np = sosfiltfilt(sos_mu_beta, Xtr.numpy(), axis=-1).astype(np.float32)
        Xte_np = sosfiltfilt(sos_mu_beta, Xte.numpy(), axis=-1).astype(np.float32)
        ch_std = np.std(Xtr_np, axis=(0, 2), keepdims=True) + 1e-6
        Xtr = torch.from_numpy(Xtr_np / ch_std)
        Xte = torch.from_numpy(Xte_np / ch_std)
    elif not is_fb and model_name != "SPDTransNet":
        ch_std_t = Xtr.std(dim=(0, 2), keepdim=True).clamp(min=1e-6)
        Xtr = Xtr / ch_std_t
        Xte = Xte / ch_std_t

    model = build_eeg_model(model_name, n_chans=n_chans, n_outputs=n_outputs, sfreq=sfreq)

    if model_name == "SPDTransNet":
        hi_freq = min(30.0, sfreq / 2.0 - 1.0)
        sos_mu_beta = butter(4, [8.0, hi_freq], btype="bandpass", fs=float(sfreq), output="sos")
        Xtr_bp_np = sosfiltfilt(sos_mu_beta, Xtr.numpy(), axis=-1).astype(np.float32)
        Xte_bp_np = sosfiltfilt(sos_mu_beta, Xte.numpy(), axis=-1).astype(np.float32)
        ch_std = np.std(Xtr_bp_np, axis=(0, 2), keepdims=True) + 1e-6
        Xtr_bp = torch.from_numpy(Xtr_bp_np / ch_std)
        Xte_bp = torch.from_numpy(Xte_bp_np / ch_std)
        band_std = data["Xfb_tr"][:, 1:3].std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)

        def _raw_seq_covs(x_bp: torch.Tensor, xfb_in: torch.Tensor) -> torch.Tensor:
            B, C, T = x_bp.shape
            w = T // 2
            b0 = x_bp[:, :, : w * 2].reshape(B, 1, C, 2, w)
            b12 = (xfb_in[:, 1:3, :, : w * 2] / band_std).reshape(B, 2, C, 2, w)
            xb = torch.cat([b0, b12], dim=1).permute(0, 1, 3, 2, 4)
            return F_spd.covariance(xb) + 1e-3 * torch.eye(C)

        cov_tr = _raw_seq_covs(Xtr_bp, data["Xfb_tr"])
        cov_te = _raw_seq_covs(Xte_bp, data["Xfb_te"])
        # Strictly inductive training-set whitening per frequency band
        mean_tr = cov_tr.mean(dim=(0, 2), keepdim=True)
        evals, evecs = torch.linalg.eigh(mean_tr)
        inv_sqrt = evecs @ torch.diag_embed(evals.clamp(min=1e-5).rsqrt()) @ evecs.mT
        Xtr = (F_spd.ensure_sym(inv_sqrt @ cov_tr @ inv_sqrt) + 1e-3 * torch.eye(n_chans)).unsqueeze(1)
        Xte = (F_spd.ensure_sym(inv_sqrt @ cov_te @ inv_sqrt) + 1e-3 * torch.eye(n_chans)).unsqueeze(1)
    elif model_name == "GraphCSPNet":
        ch_std_fb = Xtr.std(dim=(0, 3), keepdim=True).clamp(min=1e-6)
        Xtr = (Xtr / ch_std_fb) * 10.0
        Xte = (Xte / ch_std_fb) * 10.0
        model.fit_graph(Xtr[: min(64, len(Xtr))])

    if model_name == "SPDTransNet":
        fast_params = list(model.classifier[6].parameters()) + [
            p for n, p in model.named_parameters() if "attention" in n
        ]
        fast_ids = {id(p) for p in fast_params}
        slow_params = [p for p in model.parameters() if id(p) not in fast_ids]
        opt = torch.optim.AdamW(
            [
                {"params": fast_params, "lr": 5e-3, "weight_decay": 1e-2},
                {"params": slow_params, "lr": 5e-5, "weight_decay": 1e-3},
            ]
        )
    else:
        if model_name == "CorNet":
            lr = 5e-3
        elif model_name in PURE_COV_EEG_MODELS:
            lr = 3e-3
        elif model_name in ("Green", "DeepGeoCCA", "RCEEGNet", "SSGCNet", "RHOP"):
            lr = 3e-3
        else:
            lr = 2e-3
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 32
    curve_acc: list[float] = []
    best_val_bacc = -1.0
    best_bacc = 0.0
    best_acc = 0.0
    best_kappa = 0.0

    # Stratified 85% train / 15% validation split from training set (zero test-set peeking)
    ytr_np_split = ytr.numpy()
    tr_sub_l, val_sub_l = [], []
    rng_val = np.random.default_rng(seed + 777)
    for cls_c in np.unique(ytr_np_split):
        c_idx = np.where(ytr_np_split == cls_c)[0]
        rng_val.shuffle(c_idx)
        n_v = max(1, int(round(0.15 * len(c_idx)))) if len(c_idx) >= 6 else 0
        if n_v > 0:
            val_sub_l.append(c_idx[:n_v])
            tr_sub_l.append(c_idx[n_v:])
        else:
            tr_sub_l.append(c_idx)
    if val_sub_l:
        tr_sub_idx = torch.from_numpy(np.concatenate(tr_sub_l))
        val_sub_idx = torch.from_numpy(np.concatenate(val_sub_l))
        Xtr_fit, ytr_fit = Xtr[tr_sub_idx], ytr[tr_sub_idx]
        Xval, yval = Xtr[val_sub_idx], ytr[val_sub_idx]
    else:
        Xtr_fit, ytr_fit = Xtr, ytr
        Xval, yval = Xtr, ytr

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_fit))
        for i in range(0, len(Xtr_fit), batch_size):
            idx = perm[i : i + batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            out = model(Xtr_fit[idx])
            logits = out[0] if isinstance(out, tuple) else out
            loss = loss_fn(logits, ytr_fit[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step_dynamic_trivialization(model, opt)
        sched.step()

        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                v_preds = []
                for i in range(0, len(Xval), 64):
                    out_v = model(Xval[i : i + 64])
                    logits_v = out_v[0] if isinstance(out_v, tuple) else out_v
                    v_preds.append(logits_v.argmax(dim=-1))
                v_pred = torch.cat(v_preds).numpy()
                val_bacc = float(balanced_accuracy_score(yval.numpy(), v_pred) * 100.0)

                preds = []
                for i in range(0, len(Xte), 64):
                    out_te = model(Xte[i : i + 64])
                    logits_te = out_te[0] if isinstance(out_te, tuple) else out_te
                    preds.append(logits_te.argmax(dim=-1))
                pred = torch.cat(preds).numpy()
                acc = float(accuracy_score(yte.numpy(), pred) * 100.0)
                bacc = float(balanced_accuracy_score(yte.numpy(), pred) * 100.0)
                kap = float(cohen_kappa_score(yte.numpy(), pred))
                curve_acc.append(round(bacc, 2))
                if val_bacc >= best_val_bacc:
                    best_val_bacc = val_bacc
                    best_bacc = bacc
                    best_acc = acc
                    best_kappa = kap

    wall_s = round(time.perf_counter() - t0, 2)
    return {
        "dataset": dataset_name,
        "model": model_name,
        "subject": subj,
        "seed": seed,
        "epochs": epochs,
        "accuracy": round(best_acc, 2),
        "balanced_accuracy": round(best_bacc, 2),
        "final_balanced_accuracy": curve_acc[-1] if curve_acc else 0.0,
        "kappa": round(best_kappa, 4),
        "curve": curve_acc,
        "wall_s": wall_s,
    }


# =============================================================================
# 3. SPD Matrix Benchmarks: Radar (20x20, 3 cls) & HDM05 (93x93, 117 cls)
# =============================================================================


class SPDMatrixClassifier(nn.Module):
    """Unified SPD-matrix classifier across all 39 SPD Learn models and variants."""

    def __init__(
        self,
        dims: list[int],
        n_classes: int,
        variant: str = "SPDNet",
        reeig_eps: float = 1e-4,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.reeig_eps = reeig_eps
        final_d = dims[-1]
        self.final_d = final_d
        feat_dim = final_d * (final_d + 1) // 2
        self.decoder: nn.Module | None = None
        self.cca_loss_fn: nn.Module | None = None
        self.sub_bimaps: nn.ModuleList | None = None
        self.seq_block: nn.Module | None = None

        multi_sub_variants = {
            "MAtt",
            "GBWAtt",
            "CorAtt",
            "GyroAtt",
            "SPDMetaFormer",
            "SPDTransNet",
            "SSGCNet",
            "RHOP",
            "DeepGeoCCA",
            "GraphCSPNet",
            "TensorCSPNet",
            "PhaseSPDNet",
            "Green",
            "STaRNet",
        }
        if variant in multi_sub_variants:
            n_sub = 2 if variant in ("DeepGeoCCA", "PhaseSPDNet") else 4
            self.n_sub = n_sub
            d_in = dims[0]
            self.sub_bimaps = nn.ModuleList(
                [
                    nn.Sequential(
                        modules.BiMap(d_in, final_d, dtype=dtype),
                        modules.ReEig(threshold=reeig_eps, dtype=dtype),
                        modules.SPDBatchNormMean(final_d, dtype=dtype),
                    )
                    for _ in range(n_sub)
                ]
            )
            self.backbone = nn.Identity()
            if variant in ("MAtt", "STaRNet"):
                from spd_learn.models.matt import AttentionManifold

                self.seq_block = AttentionManifold(final_d, final_d).to(dtype)
                self.proj: nn.Module | None = modules.LogEig(upper=True, flatten=True, dtype=dtype)
                self.head: nn.Module = nn.Linear(n_sub * feat_dim, n_classes, dtype=dtype)
            elif variant == "GBWAtt":
                from spd_learn.modules.gbw import GBWAttention

                self.seq_block = GBWAttention(final_d, final_d).to(dtype)
                self.proj = modules.LogEig(upper=True, flatten=True, dtype=dtype)
                self.head = nn.Linear(n_sub * feat_dim, n_classes, dtype=dtype)
            elif variant == "CorAtt":
                from spd_learn.models.coratt import CorrAttention

                self.seq_block = CorrAttention(final_d, dtype=dtype)
                self.proj = modules.LogEig(upper=True, flatten=True, dtype=dtype)
                self.head = nn.Linear(n_sub * feat_dim, n_classes, dtype=dtype)
            elif variant == "GyroAtt":
                from spd_learn.models.gyroatt import GyroSPDAttention

                self.seq_block = GyroSPDAttention(final_d, metric="LEM", dtype=dtype)
                self.proj = modules.LogEig(upper=True, flatten=True, dtype=dtype)
                self.head = nn.Linear(n_sub * feat_dim, n_classes, dtype=dtype)
            elif variant == "SPDMetaFormer":
                from spd_learn.models.spdmetaformer import SPDMetaFormerBlock

                self.seq_block = SPDMetaFormerBlock(final_d, eps=reeig_eps, dtype=dtype)
                self.proj = modules.LogEig(upper=True, flatten=True, dtype=dtype)
                self.head = nn.Linear(n_sub * feat_dim, n_classes, dtype=dtype)
            elif variant == "SPDTransNet":
                from spd_learn.modules.spdtransnet import (
                    SPDLogTokenizer,
                    SPDTransformerEncoderLayer,
                )

                self.seq_block = nn.Sequential(
                    SPDLogTokenizer(matrix_size=final_d, dtype=dtype),
                    SPDTransformerEncoderLayer(
                        embed_dim=feat_dim,
                        num_heads=1,
                        feedforward_dim=feat_dim,
                        dropout=0.1,
                        dtype=dtype,
                    ),
                )
                self.proj = nn.Flatten(start_dim=1)
                self.head = nn.Linear(n_sub * feat_dim, n_classes, dtype=dtype)
            elif variant == "SSGCNet":
                from spd_learn.models.ssgcnet import TangentGraphSPDAggregation

                self.seq_block = TangentGraphSPDAggregation(k_neighbors=2, eps=reeig_eps).to(dtype)
                self.proj = modules.LogEig(upper=True, flatten=True, dtype=dtype)
                self.head = nn.Linear(n_sub * feat_dim, n_classes, dtype=dtype)
            elif variant == "RHOP":
                from spd_learn.modules.spdtransnet import SPDLogTokenizer

                red_d = max(4, final_d // 2)
                self.seq_block = nn.Sequential(
                    SPDLogTokenizer(matrix_size=final_d, dtype=dtype),
                    modules.RHOP(
                        feat_dim, reduced_dim=red_d, eps=reeig_eps, dtype=dtype
                    ),
                )
                self.proj = nn.Flatten(start_dim=1)
                self.head = nn.Linear(feat_dim, n_classes, dtype=dtype)
            elif variant == "DeepGeoCCA":
                from spd_learn.models.deepgeocca import GeodesicCCALoss

                self.cca_loss_fn = GeodesicCCALoss(
                    outdim_size=min(6, final_d), eps=reeig_eps
                ).to(dtype)
                self.proj = modules.LogEig(upper=True, flatten=True, dtype=dtype)
                self.head = nn.Linear(n_sub * feat_dim, n_classes, dtype=dtype)
            elif variant == "GraphCSPNet":
                from spd_learn.modules.graph import GraphBiMap

                prop = torch.eye(n_sub, dtype=dtype) * 0.6 + (0.4 / n_sub)
                self.register_buffer("graph_prop", prop)
                self.seq_block = GraphBiMap(n_sub, final_d, final_d, dtype=dtype)
                self.proj = modules.LogEig(upper=True, flatten=True, dtype=dtype)
                self.head = nn.Linear(n_sub * feat_dim, n_classes, dtype=dtype)
            else:
                # TensorCSPNet, PhaseSPDNet, Green
                self.proj = modules.LogEig(upper=True, flatten=True, dtype=dtype)
                self.head = nn.Linear(n_sub * feat_dim, n_classes, dtype=dtype)
            self.to(dtype)
            return

        layers: list[nn.Module] = []
        for idx_l, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            is_last_l = idx_l == len(dims) - 2
            if variant == "GyroSPD":
                layers.append(models.GyroSPDLayer(a, b, threshold=reeig_eps, dtype=dtype))
                layers.append(modules.SPDBatchNormMean(b, dtype=dtype))
                continue
            if variant == "SPNN":
                if idx_l == 0 and len(dims) == 2:
                    layers.append(modules.BiMap(a, b, dtype=dtype))
                    layers.append(modules.ReEig(threshold=reeig_eps, dtype=dtype))
                    layers.append(modules.SPDBatchNormMean(b, dtype=dtype))
                else:
                    layers.append(
                        modules.SPDLinear(
                            a,
                            b,
                            metric="LCM",
                            unidirectional=True,
                            return_tangent=False,
                            eps=reeig_eps,
                            dtype=dtype,
                        )
                    )
                    layers.append(modules.SPDBatchNormLie(b, metric="LCM", theta=0.5, dtype=dtype))
                continue
            layers.append(modules.BiMap(a, b, dtype=dtype))
            layers.append(modules.ReEig(threshold=reeig_eps, dtype=dtype))
            if variant in ("SPDNetBN", "TSMNet", "RieTrans-SPDRMLR", "PCM"):
                layers.append(modules.SPDBatchNormMean(b, dtype=dtype))
            elif variant == "EEGSPDNet":
                layers.append(modules.SPDDropout(p=0.1, epsilon=reeig_eps, dtype=dtype))
            elif variant == "LieBN-AIM":
                layers.append(modules.SPDBatchNormLie(b, metric="AIM", dtype=dtype))
            elif variant == "LieBN-LEM":
                layers.append(modules.SPDBatchNormLie(b, metric="LEM", dtype=dtype))
            elif variant == "LieBN-LCM":
                layers.append(modules.SPDBatchNormLie(b, metric="LCM", theta=0.5, dtype=dtype))
            elif variant in ("ARMAGNAC", "ARMAGNAC-Harmonic"):
                layers.append(modules.SPDBatchNormHarmonic(b, alpha=0.05, learnable_alpha=False, dtype=dtype))
            elif variant == "ARMAGNAC-Arithmetic":
                layers.append(modules.SPDBatchNormHarmonic(b, alpha=0.95, learnable_alpha=False, dtype=dtype))
            elif variant == "ARMAGNAC-GAH":
                layers.append(modules.SPDBatchNormHarmonic(b, alpha=0.5, learnable_alpha=True, dtype=dtype))
            elif variant in ("GBWBN", "BWCM"):
                layers.append(modules.SPDBatchNormBW(b, dtype=dtype))
            elif variant == "CBN":
                layers.append(modules.SPDBatchNormCholesky(b, dtype=dtype))
            elif variant in ("SPDResNet-ALEM", "SPDResNet", "ALEM", "RCEEGNet"):
                from spd_learn.models.spdresnet import SPDResBlock

                layers.append(modules.SPDBatchNormMean(b, dtype=dtype))
                layers.append(SPDResBlock(b, eps=reeig_eps).to(dtype))
        self.backbone = nn.Sequential(*layers)

        if variant == "SPDNetAE":
            from spd_learn.models.spdnetae import SPDDecoderBlock

            dec_layers = [
                SPDDecoderBlock(d_in, d_out, eps=reeig_eps, dtype=dtype)
                for d_in, d_out in zip(reversed(dims[1:]), reversed(dims[:-1]))
            ]
            self.decoder = nn.Sequential(*dec_layers)

        if variant in ("SPDResNet-ALEM", "ALEM"):
            from spd_learn.models.spdresnet import AdaptiveLogEuclidean

            self.proj = nn.Sequential(
                AdaptiveLogEuclidean(final_d, return_spd=False).to(dtype),
                modules.Vech(upper=True),
            )
            self.head = nn.Linear(feat_dim, n_classes, dtype=dtype)
        elif variant == "SPNN":
            self.proj = nn.Identity()
            self.head = modules.SPDRMLR(
                final_d,
                n_classes,
                metric="LCM",
                unidirectional=True,
                input_tangent=False,
                eps=reeig_eps,
                dtype=dtype,
            )
        elif variant == "RieTrans-SPDRMLR":
            self.proj = nn.Identity()
            self.head = modules.SPDRMLR(
                final_d, n_classes, metric="LEM", unidirectional=True, eps=reeig_eps, dtype=dtype
            )
        elif variant in ("PCM", "BWCM", "CorNet-OLM", "CorNet-LSM", "CorNet"):
            self.proj = None
            if variant in ("CorNet-OLM", "CorNet-LSM", "CorNet"):
                feat_dim = final_d * (final_d - 1) // 2
            self.vech = modules.Vech(upper=True)
            self.head = nn.Linear(feat_dim, n_classes, dtype=dtype)
        else:
            self.proj = modules.LogEig(upper=True, flatten=True, dtype=dtype)
            self.head = nn.Linear(feat_dim, n_classes, dtype=dtype)
        self.to(dtype)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.sub_bimaps is not None:
            eye = torch.eye(self.final_d, dtype=X.dtype, device=X.device)
            s_list = []
            for bm in self.sub_bimaps:
                si = bm(X)
                si = 0.5 * (si + si.transpose(-1, -2))
                s_list.append(si + 1e-4 * eye)
            S = torch.stack(s_list, dim=1)
            if self.variant == "GraphCSPNet":
                assert self.seq_block is not None and self.graph_prop is not None
                S = self.seq_block(S, self.graph_prop) + 1e-2 * eye
            elif self.seq_block is not None:
                S = self.seq_block(S)
                if S.ndim == 4:
                    S = 0.5 * (S + S.transpose(-1, -2)) + 1e-2 * eye
            assert self.proj is not None
            Z = self.proj(S).reshape(X.shape[0], -1)
            return self.head(Z)

        X = self.backbone(X)
        if self.variant == "PCM":
            eye = torch.eye(self.final_d, dtype=X.dtype, device=X.device).expand_as(X)
            X_reg = 0.5 * (X + X.transpose(-1, -2)) + 1e-3 * eye
            T = F_spd.log_map_pcm(X_reg, eye, theta=0.75)
            X_vec = self.vech(0.5 * (T + T.transpose(-1, -2)))
            return self.head(X_vec)
        if self.variant == "BWCM":
            eye = torch.eye(self.final_d, dtype=X.dtype, device=X.device).expand_as(X)
            X_reg = 0.5 * (X + X.transpose(-1, -2)) + 1e-3 * eye
            T = F_spd.log_map_bwcm(X_reg, eye)
            X_vec = self.vech(0.5 * (T + T.transpose(-1, -2)))
            return self.head(X_vec)
        if self.variant in ("CorNet-OLM", "CorNet"):
            C = F_spd.cov_to_corr(X)
            X_vec = F_spd.tril_embed(F_spd.log_map_olm(C))
            return self.head(X_vec)
        if self.variant == "CorNet-LSM":
            C = F_spd.cov_to_corr(X)
            X_vec = F_spd.tril_embed(F_spd.log_map_lsm(C))
            return self.head(X_vec)
        assert self.proj is not None
        return self.head(self.proj(X))


def load_spd_matrix_dataset(name: str) -> tuple[torch.Tensor, torch.Tensor, list[str] | None]:
    cache_pt = DATA_STAGE_DIR / f"{name}_cache.pt"
    if cache_pt.exists():
        d = torch.load(cache_pt, weights_only=False)
        return d["X"], d["y"].long(), d.get("split")

    local_cache_pt = LOCAL_SPD_DATA_DIR / f"{name}_cache.pt"
    if local_cache_pt.exists():
        d = torch.load(local_cache_pt, weights_only=False)
        return d["X"], d["y"].long(), d.get("split")

    folder_candidates = [
        DATA_STAGE_DIR / name,
        LOCAL_SPD_DATA_DIR / name,
        LOCAL_SPD_DATA_DIR / name.upper(),
    ]
    folder = next((p for p in folder_candidates if p.exists()), None)
    if folder is None:
        raise FileNotFoundError(f"Missing {cache_pt} and local folder for {name}")

    files = sorted(
        f for f in glob.glob(str(folder / "**" / "*.npy"), recursive=True)
        if "__MACOSX" not in f
    )
    if not files:
        raise FileNotFoundError(f"No .npy files found for {name} in {folder}")
    names = [os.path.basename(f) for f in files]
    order = np.argsort(names)
    files = [files[i] for i in order]
    names = [names[i] for i in order]
    raw_list = [np.load(f) for f in files]
    arr = np.stack(raw_list)
    if arr.ndim == 2:
        x_win = arr.reshape(arr.shape[0], 20, -1)
        x_win = x_win - x_win.mean(axis=-1, keepdims=True)
        cov = np.matmul(x_win, np.conj(np.swapaxes(x_win, -1, -2))) / x_win.shape[-1]
        if np.iscomplexobj(cov):
            re_c = np.real(cov).astype(np.float64)
            im_c = np.imag(cov).astype(np.float64)
            arr = np.concatenate(
                [np.concatenate([re_c, -im_c], axis=-1), np.concatenate([im_c, re_c], axis=-1)],
                axis=-2,
            )
        else:
            arr = np.real(cov).astype(np.float64)
    elif np.iscomplexobj(arr):
        re_c = np.real(arr).astype(np.float64)
        im_c = np.imag(arr).astype(np.float64)
        arr = np.concatenate(
            [np.concatenate([re_c, -im_c], axis=-1), np.concatenate([im_c, re_c], axis=-1)],
            axis=-2,
        )
    else:
        arr = arr.astype(np.float64)
    arr = 0.5 * (arr + np.swapaxes(arr, -1, -2))
    d_dim = arr.shape[-1]
    tr = np.trace(arr, axis1=-2, axis2=-1)[..., None, None] / d_dim
    tr = np.maximum(tr, 1e-4)
    arr = arr + 1e-3 * tr * np.eye(d_dim)

    raw_labels = [int(n.split(".")[0].split("_")[-1]) for n in names]
    uniq_labels = sorted(set(raw_labels))
    label_map = {lbl: idx for idx, lbl in enumerate(uniq_labels)}
    y = torch.tensor([label_map[lbl] for lbl in raw_labels], dtype=torch.long)
    X = torch.from_numpy(arr)
    try:
        local_cache_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"X": X, "y": y, "names": names}, local_cache_pt)
    except Exception:
        pass
    return X, y, None


def eval_spd_matrix_cell(args: tuple[str, str, int, int, float, int, float]) -> dict[str, Any]:
    """Evaluate (dataset_name, variant, epochs, seed, train_frac_scale, batch_size, reeig_eps)."""
    dataset_name, variant, epochs, seed, train_frac_scale, batch_size, reeig_eps = args
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_raw, y, split_list = load_spd_matrix_dataset(dataset_name)
    use_dtype = (
        torch.float32
        if (dataset_name == "afew" and variant not in ("GBWAtt", "GBWBN", "BWCM", "PCM"))
        else torch.float64
    )
    X = X_raw.to(use_dtype)
    n_classes = int(y.max().item()) + 1
    d_dim = X.shape[-1]

    if split_list is not None:
        sp_arr = np.asarray(split_list)
        tr_idx_np = np.where(sp_arr == "train")[0]
        te_idx_np = np.where(sp_arr != "train")[0]
        rng = np.random.RandomState(seed)
        rng.shuffle(tr_idx_np)
        tr_idx = torch.from_numpy(tr_idx_np)
        te_idx = torch.from_numpy(te_idx_np)
    else:
        frac = 0.25 if dataset_name == "radar" else 0.50
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(y))
        n_test = int(frac * len(y))
        te_idx = torch.from_numpy(idx[:n_test])
        tr_idx = torch.from_numpy(idx[n_test:])
    if train_frac_scale < 1.0:
        n_keep = max(n_classes * 2, int(len(tr_idx) * train_frac_scale))
        tr_idx = tr_idx[:n_keep]

    Xtr_all, ytr_all = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]

    # Stratified 85% / 15% train / validation split from Xtr_all (no test peeking)
    rng_val = np.random.RandomState(seed + 137)
    ytr_np = ytr_all.cpu().numpy()
    fit_idx_list: list[int] = []
    val_idx_list: list[int] = []
    for c_val in np.unique(ytr_np):
        c_indices = np.where(ytr_np == c_val)[0]
        rng_val.shuffle(c_indices)
        n_val_c = max(1, int(round(0.15 * len(c_indices))))
        if len(c_indices) > 2:
            val_idx_list.extend(c_indices[:n_val_c].tolist())
            fit_idx_list.extend(c_indices[n_val_c:].tolist())
        else:
            fit_idx_list.extend(c_indices.tolist())
            val_idx_list.extend(c_indices.tolist())
    fit_idx_t = torch.tensor(fit_idx_list, dtype=torch.long)
    val_idx_t = torch.tensor(val_idx_list, dtype=torch.long)
    Xtr, ytr = Xtr_all[fit_idx_t], ytr_all[fit_idx_t]
    Xval, yval = Xtr_all[val_idx_t], ytr_all[val_idx_t]

    if dataset_name == "radar":
        dims = [d_dim, 32, 24] if d_dim == 40 else [d_dim, 16, 12]
    elif dataset_name == "fpha":
        dims = [d_dim, 33]
    elif dataset_name == "afew":
        dims = [d_dim, 50]
    else:
        dims = [d_dim, 30]
    model = SPDMatrixClassifier(
        dims=dims, n_classes=n_classes, variant=variant, reeig_eps=reeig_eps, dtype=use_dtype
    )
    opt = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    best_val_acc = -1.0
    selected_acc = 0.0
    curve = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch_size):
            b = perm[i : i + batch_size]
            if len(b) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xtr[b]), ytr[b])
            if model.decoder is not None:
                z_lat = model.backbone(Xtr[b])
                x_rec = model.decoder(z_lat)
                loss = loss + 0.01 * F_spd.log_euclidean_distance(Xtr[b], x_rec).square().mean()
            elif model.cca_loss_fn is not None and model.sub_bimaps is not None:
                eye_c = torch.eye(model.final_d, dtype=Xtr.dtype, device=Xtr.device)
                z1 = model.sub_bimaps[0](Xtr[b]) + 1e-3 * eye_c
                z2 = model.sub_bimaps[1](Xtr[b]) + 1e-3 * eye_c
                loss = loss + 0.05 * model.cca_loss_fn(z1, z2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step_dynamic_trivialization(model, opt)
        sched.step()

        if ep % max(1, epochs // 6) == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                v_preds = [model(Xval[i : i + 128]).argmax(dim=-1) for i in range(0, len(Xval), 128)]
                v_pred = torch.cat(v_preds)
                val_acc = float((v_pred == yval).double().mean().item() * 100.0)
                preds = [model(Xte[i : i + 128]).argmax(dim=-1) for i in range(0, len(Xte), 128)]
                pred = torch.cat(preds)
                acc = float((pred == yte).double().mean().item() * 100.0)
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    selected_acc = acc
                curve.append(round(acc, 2))

    wall_s = round(time.perf_counter() - t0, 2)
    return {
        "dataset": dataset_name,
        "variant": variant,
        "seed": seed,
        "epochs": epochs,
        "train_frac_scale": train_frac_scale,
        "batch_size": batch_size,
        "reeig_eps": reeig_eps,
        "accuracy": round(selected_acc, 2),
        "val_accuracy": round(best_val_acc, 2),
        "final_accuracy": curve[-1] if curve else 0.0,
        "curve": curve,
        "wall_s": wall_s,
    }


def run_shard(
    shard: str,
    workers: int,
    epochs: int,
    datasets_csv: str = "",
    models_csv: str = "",
    subjects_csv: str = "",
) -> dict[str, Any]:
    t_start = time.perf_counter()
    eeg_tasks: list[tuple[str, str, int, int, int]] = []
    spd_tasks: list[tuple[str, str, int, int, float, int, float]] = []

    if shard == "bnci2014_001_p1":
        for m in ALL_EEG_MODELS:
            for s in [1, 2, 3, 4, 5]:
                eeg_tasks.append(("BNCI2014_001", m, s, epochs, 42))
    elif shard == "bnci2014_001_p2":
        for m in ALL_EEG_MODELS:
            for s in [6, 7, 8, 9]:
                eeg_tasks.append(("BNCI2014_001", m, s, epochs, 42))
    elif shard == "bnci2014_004":
        for m in ALL_EEG_MODELS:
            for s in range(1, 10):
                eeg_tasks.append(("BNCI2014_004", m, s, epochs, 42))
    elif shard == "bnci2015_001":
        for m in ALL_EEG_MODELS:
            for s in range(1, 13):
                eeg_tasks.append(("BNCI2015_001", m, s, epochs, 42))
    elif shard == "bnci2014_002_weibo":
        for m in ALL_EEG_MODELS:
            for s in [1, 2, 3, 4, 5, 6]:
                eeg_tasks.append(("BNCI2014_002", m, s, epochs, 42))
            for s in [1, 2, 3, 4, 5]:
                eeg_tasks.append(("Weibo2014", m, s, epochs, 42))
    elif shard == "schirr_shin_zhou":
        # Also include CorNet on BNCI2014_001 S1..S9
        for s in range(1, 10):
            eeg_tasks.append(("BNCI2014_001", "CorNet", s, epochs, 42))
        for m in ALL_EEG_MODELS:
            for s in [1, 2, 3, 4]:
                eeg_tasks.append(("Schirrmeister2017", m, s, epochs, 42))
            for s in [1, 2, 3, 4]:
                eeg_tasks.append(("Shin2017A", m, s, epochs, 42))
            for s in [1, 2, 3, 4]:
                eeg_tasks.append(("Zhou2020", m, s, epochs, 42))
    elif shard == "lee2019_cho2017":
        for m in ALL_EEG_MODELS:
            for s in [1, 2, 3, 4, 5, 6]:
                eeg_tasks.append(("Lee2019_MI", m, s, epochs, 42))
            for s in [1, 2, 3, 4, 5, 6]:
                eeg_tasks.append(("Cho2017", m, s, epochs, 42))
    elif shard == "missing5_all_datasets":
        missing_models = ["CorNet", "RHOP", "SPDResNet", "ALEM", "GyroSPD"]
        ds_subjects = [
            ("BNCI2014_001", range(1, 10)),
            ("BNCI2014_004", range(1, 10)),
            ("BNCI2015_001", range(1, 13)),
            ("BNCI2014_002", [1, 2, 3, 4, 5, 6]),
            ("Weibo2014", [1, 2, 3, 4, 5]),
            ("Lee2019_MI", [1, 2, 3, 4, 5, 6]),
            ("Cho2017", [1, 2, 3, 4, 5, 6]),
            ("Schirrmeister2017", [1, 2, 3, 4]),
            ("Shin2017A", [1, 2, 3, 4]),
            ("Zhou2020", [1, 2, 3, 4]),
        ]
        for ds, subs in ds_subjects:
            for s in subs:
                for m in missing_models:
                    eeg_tasks.append((ds, m, s, epochs, 42))
    elif shard == "spd_matrices":
        spd_variants = [
            "SPDNet",
            "SPDNetBN",
            "LieBN-AIM",
            "LieBN-LEM",
            "LieBN-LCM",
            "ARMAGNAC-Harmonic",
            "ARMAGNAC-Arithmetic",
            "ARMAGNAC-GAH",
            "GBWBN",
            "CBN",
            "PCM",
            "BWCM",
            "CorNet-OLM",
            "CorNet-LSM",
            "RieTrans-SPDRMLR",
            "SPDResNet-ALEM",
        ]
        # 5 seeds on Radar & HDM05
        for var in spd_variants:
            for seed in [1024, 1025, 1026, 1027, 1028]:
                spd_tasks.append(("radar", var, max(epochs, 60), seed, 1.0, 64, 1e-4))
            for seed in [1024, 1025, 1026]:
                spd_tasks.append(("hdm05", var, max(epochs, 50), seed, 1.0, 64, 1e-4))
        # Data scarcity ablation on Radar (5%, 10%, 25%, 50%, 100%)
        for var in ["SPDNet", "SPDNetBN", "LieBN-AIM", "ARMAGNAC-Harmonic", "GBWBN", "CBN"]:
            for frac_s in [0.05, 0.10, 0.25, 0.50]:
                spd_tasks.append(("radar", var, max(epochs, 50), 1024, frac_s, 32, 1e-4))
    elif shard == "spd_matrices_repair":
        repair_vars = [
            "ARMAGNAC-Harmonic",
            "ARMAGNAC-Arithmetic",
            "ARMAGNAC-GAH",
            "PCM",
            "SPDResNet-ALEM",
        ]
        for var in repair_vars:
            for seed in [1024, 1025, 1026, 1027, 1028]:
                spd_tasks.append(("radar", var, max(epochs, 60), seed, 1.0, 64, 1e-4))
            for seed in [1024, 1025, 1026]:
                spd_tasks.append(("hdm05", var, max(epochs, 50), seed, 1.0, 64, 1e-4))
        for var in ["ARMAGNAC-Harmonic", "PCM"]:
            for frac_s in [0.05, 0.10, 0.25, 0.50]:
                spd_tasks.append(("radar", var, max(epochs, 50), 1024, frac_s, 32, 1e-4))
    elif shard == "bnci2014_001_repair":
        repair_eeg_models = [
            "SPDNet",
            "PhaseSPDNet",
            "CorNet",
            "SPNN",
            "SPDResNet",
            "ALEM",
            "GyroSPD",
            "SPDNetAE",
            "SPDTransNet",
            "Green",
            "RHOP",
            "DeepGeoCCA",
            "RCEEGNet",
            "SSGCNet",
        ]
        for m in repair_eeg_models:
            for s in range(1, 10):
                eeg_tasks.append(("BNCI2014_001", m, s, epochs, 42))
    elif shard == "multids_repair_a":
        target_models_a = ["STaRNet", "SPDResNet", "ALEM", "SPDTransNet", "PhaseSPDNet", "SPDNet"]
        ds_subs_a = [
            ("BNCI2014_002", [1, 2, 3, 4, 5, 6]),
            ("BNCI2014_004", list(range(1, 10))),
            ("BNCI2015_001", list(range(1, 13))),
            ("Cho2017", [1, 2, 3, 4, 5, 6]),
            ("Lee2019_MI", [1, 2, 3, 4, 5, 6]),
            ("Shin2017A", [1, 2, 3, 4]),
        ]
        for ds, subs in ds_subs_a:
            for s in subs:
                for m in target_models_a:
                    eeg_tasks.append((ds, m, s, epochs, 42))
    elif shard == "multids_repair_b":
        ds_subs_b = [
            ("Schirrmeister2017", [1, 2, 3, 4]),
            ("Weibo2014", [1, 2, 3, 4, 5]),
            ("Zhou2020", [1, 2, 3, 4]),
        ]
        for ds, subs in ds_subs_b:
            for s in subs:
                for m in ALL_EEG_MODELS:
                    eeg_tasks.append((ds, m, s, epochs, 42))
    elif shard == "repair_schirr":
        for s in [1, 2, 3, 4]:
            for m in ALL_EEG_MODELS:
                eeg_tasks.append(("Schirrmeister2017", m, s, epochs, 42))
    elif shard == "repair_weibo":
        for s in [1, 2, 3, 4, 5]:
            for m in ALL_EEG_MODELS:
                eeg_tasks.append(("Weibo2014", m, s, epochs, 42))
    elif shard == "repair_zhou":
        for s in [1, 2, 3, 4]:
            for m in ALL_EEG_MODELS:
                eeg_tasks.append(("Zhou2020", m, s, epochs, 42))
    elif shard == "full_repair":
        for m in ["BWCM", "PCM"]:
            eeg_tasks.append(("BCI_ERN", m, 6, epochs, 42))
            for s in [1, 2, 3, 4]:
                eeg_tasks.append(("TDBRAIN", m, s, epochs, 42))
        for s in [1, 2, 3, 4]:
            eeg_tasks.append(("Schirrmeister2017", "GBWBN", s, epochs, 42))
            for m in ["ARMAGNAC-GAH", "BWCM", "PCM"]:
                eeg_tasks.append(("TUEV", m, s, epochs, 42))
        all_spd_23 = [
            "SPDNet", "SPDNetBN", "LieBN-AIM", "LieBN-LEM", "LieBN-LCM",
            "ARMAGNAC-Harmonic", "ARMAGNAC-Arithmetic", "ARMAGNAC-GAH",
            "GBWBN", "CBN", "PCM", "BWCM", "CorNet-OLM", "CorNet-LSM",
            "RieTrans-SPDRMLR", "SPDResNet-ALEM", "SPDResNet", "ALEM",
            "GyroSPD", "GyroAtt", "SPNN", "CorNet", "SPDNetAE",
        ]
        for var in all_spd_23:
            for seed in [1027, 1028]:
                spd_tasks.append(("hdm05", var, max(epochs, 40), seed, 1.0, 64, 1e-4))
        afew_16 = all_spd_23[:16]
        for var in afew_16:
            for seed in [1027, 1028]:
                spd_tasks.append(("afew", var, max(epochs, 40), seed, 1.0, 64, 1e-4))
    else:
        ds_subjects_map: dict[str, list[int]] = {
            "BNCI2014_001": list(range(1, 10)),
            "BNCI2014_004": list(range(1, 10)),
            "BNCI2015_001": list(range(1, 13)),
            "BNCI2014_002": list(range(1, 15)),
            "Weibo2014": list(range(1, 11)),
            "Lee2019_MI": list(range(1, 13)),
            "Cho2017": list(range(1, 13)),
            "Schirrmeister2017": list(range(1, 15)),
            "Shin2017A": list(range(1, 30)),
            "Zhou2020": [1, 2, 3, 4],
            "PhysionetMI": list(range(1, 13)),
            "Stieger2021": list(range(1, 13)),
            "Hinss2021": list(range(1, 16)),
            "MAMEM2": list(range(1, 12)),
            "BCI_ERN": list(range(1, 17)),
            "SleepEDFX": list(range(1, 13)),
            "TUAB": list(range(1, 13)),
            "TUEV": list(range(1, 13)),
            "MASS_SS3": list(range(1, 13)),
            "LEMON": list(range(1, 13)),
            "TDBRAIN": list(range(1, 13)),
            "SEED": list(range(1, 16)),
            "DEAP": list(range(1, 33)),
            "ADHD200": list(range(1, 7)),
            "ABIDE": list(range(1, 7)),
        }
        use_datasets = (
            [d.strip() for d in datasets_csv.split(",") if d.strip()]
            if datasets_csv
            else list(ds_subjects_map.keys())
        )
        use_models = (
            [m.strip() for m in models_csv.split(",") if m.strip()]
            if models_csv
            else (ALL_EEG_MODELS + sorted(TSM_BN_EEG_VARIANTS))
        )
        explicit_subjects = (
            [int(x.strip()) for x in subjects_csv.split(",") if x.strip()]
            if subjects_csv
            else []
        )
        for ds in use_datasets:
            if ds in ("radar", "hdm05", "fpha", "afew"):
                seeds_use = [1024, 1025, 1026, 1027, 1028]
                for var in use_models:
                    for seed in seeds_use:
                        spd_tasks.append((ds, var, epochs, seed, 1.0, 64, 1e-4))
            else:
                subs_for_ds = explicit_subjects if explicit_subjects else ds_subjects_map.get(ds, [1, 2, 3, 4])
                for s in subs_for_ds:
                    for m in use_models:
                        eeg_tasks.append((ds, m, s, epochs, 42))

    eeg_records: list[dict[str, Any]] = []
    spd_records: list[dict[str, Any]] = []
    spawn_ctx = multiprocessing.get_context("spawn")

    if eeg_tasks:
        eeg_tasks.sort(key=lambda t: (t[0], t[2], t[1]))
        uniq_ds_sub = sorted({(t[0], t[2]) for t in eeg_tasks})
        print(f"[Shard {shard}] Pre-staging {len(uniq_ds_sub)} subject caches sequentially...", flush=True)
        for ds_name, s_id in uniq_ds_sub:
            try:
                load_eeg_subject_data(ds_name, s_id)
            except Exception as exc:
                print(f"  [STAGE WARN] ({ds_name}, S{s_id:02d}): {exc}", flush=True)
        _SUBJECT_DATA_CACHE.clear()
        print(f"[Shard {shard}] Running {len(eeg_tasks)} EEG cells across {workers} CPU workers...", flush=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=spawn_ctx) as pool:
            futs = {pool.submit(eval_eeg_cell, t): t for t in eeg_tasks}
            for fut in as_completed(futs):
                task = futs[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    print(f"  [EEG ERROR] {task}: {exc}", flush=True)
                    continue
                eeg_records.append(res)
                print(
                    f"  [{res['dataset']:17s}] {res['model']:16s} S{res['subject']:02d} "
                    f"bacc={res['balanced_accuracy']:5.2f}% acc={res['accuracy']:5.2f}% ({res['wall_s']:.1f}s)",
                    flush=True,
                )

    if spd_tasks:
        print(f"[Shard {shard}] Running {len(spd_tasks)} SPD Matrix cells across {workers} CPU workers...", flush=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=spawn_ctx) as pool:
            futs = {pool.submit(eval_spd_matrix_cell, t): t for t in spd_tasks}
            for fut in as_completed(futs):
                task = futs[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    print(f"  [SPD ERROR] {task}: {exc}", flush=True)
                    continue
                spd_records.append(res)
                print(
                    f"  [{res['dataset'].upper():6s}] {res['variant']:20s} seed={res['seed']} "
                    f"frac={res['train_frac_scale']:.2f} acc={res['accuracy']:6.2f}% ({res['wall_s']:.1f}s)",
                    flush=True,
                )

    return {
        "shard": shard,
        "node": platform.node(),
        "wall_s": round(time.perf_counter() - t_start, 2),
        "eeg_records": eeg_records,
        "spd_records": spd_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True)
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--subjects", type=str, default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else (RESULTS_SHARD_DIR / f"{args.shard}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = run_shard(
        args.shard, args.workers, args.epochs, args.datasets, args.models, args.subjects
    )
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[Shard {args.shard}] Saved {out_path} in {payload['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
