# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Merge genuine Voyager shard outputs into repro/voyager_spdbench_results.json."""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPRO_DIR = Path(__file__).resolve().parent
SHARDS_DIR = REPRO_DIR / "shards"
RESULTS_PATH = REPRO_DIR / "voyager_spdbench_results.json"

DS_NAME_TO_KEY: dict[str, str] = {
    "BNCI2014_001": "bnci2014_001",
    "BNCI2014_002": "bnci2014_002",
    "BNCI2014_004": "bnci2014_004",
    "BNCI2015_001": "bnci2015_001",
    "Cho2017": "cho2017",
    "Lee2019_MI": "lee2019_mi",
    "Schirrmeister2017": "schirrmeister2017",
    "Shin2017A": "shin2017a",
    "Weibo2014": "weibo2014",
    "Zhou2020": "zhou2020",
    "PhysionetMI": "physionet_mi",
    "Stieger2021": "stieger2021",
    "Hinss2021": "hinss2021",
    "MAMEM2": "mamem2",
    "BCI_ERN": "bci_ern",
    "SleepEDFX": "sleep_edfx",
    "TUAB": "tuab",
    "TUEV": "tuev",
    "MASS_SS3": "mass_ss3",
    "LEMON": "lemon",
    "TDBRAIN": "tdbrain",
    "SEED": "seed",
    "DEAP": "deap",
    "ADHD200": "adhd200",
    "ABIDE": "abide",
}


def merge_gen_shards() -> None:
    res: dict[str, Any] = json.loads(RESULTS_PATH.read_text())
    eeg_ds: dict[str, dict[str, Any]] = res.setdefault("eeg_datasets", {})

    # 1. Clear old surrogate entries for datasets replaced with genuine data
    replaced_eeg_keys = ["mass_ss3", "lemon", "tdbrain", "abide", "adhd200", "deap"]
    if (SHARDS_DIR / "gen_seed_v2.json").exists():
        replaced_eeg_keys.append("seed")
    for k in replaced_eeg_keys:
        eeg_ds[k] = {}
    if (SHARDS_DIR / "gen_fpha_all.json").exists():
        res["fpha"] = {}

    # 2. Collect all records from gen_*.json shards (excluding old gen_seed.json if gen_seed_v2.json exists)
    eeg_grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    spd_grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)

    shard_files = sorted(glob.glob(str(SHARDS_DIR / "gen_*.json")))
    for sp in shard_files:
        fname = Path(sp).name
        if fname == "gen_seed.json":
            continue
        payload = json.loads(Path(sp).read_text())
        for r in payload.get("eeg_records", []):
            ds_k = DS_NAME_TO_KEY.get(r["dataset"], r["dataset"].lower())
            m = r["model"]
            subj = int(r["subject"])
            eeg_grouped[(ds_k, m)][subj] = r
        for r in payload.get("spd_records", []):
            ds_k = r["dataset"].lower()
            var = r["variant"]
            seed = int(r["seed"])
            spd_grouped[(ds_k, var)][seed] = r

    # 3. Write aggregated EEG dataset x model summaries
    for (ds_k, m), subj_map in sorted(eeg_grouped.items()):
        ds_bucket = eeg_ds.setdefault(ds_k, {})
        s_ids = sorted(subj_map.keys())
        s_means = {str(s): round(float(subj_map[s]["balanced_accuracy"]), 2) for s in s_ids}
        s_finals = {
            str(s): round(float(subj_map[s].get("final_balanced_accuracy", subj_map[s]["balanced_accuracy"])), 2)
            for s in s_ids
        }
        s_kappas = {str(s): round(float(subj_map[s].get("kappa", 0.0)), 4) for s in s_ids}
        vals = np.array([float(v) for v in s_means.values()], dtype=np.float64)
        fvals = np.array([float(v) for v in s_finals.values()], dtype=np.float64)
        curves = [subj_map[s].get("curve", []) for s in s_ids if len(subj_map[s].get("curve", [])) == 6]
        mean_curve = (
            [round(float(x), 2) for x in np.mean(np.array(curves, dtype=np.float64), axis=0)]
            if curves
            else [round(float(vals.mean()), 2)] * 6
        )
        mean_v = round(float(vals.mean()), 2)
        fin_v = round(float(fvals.mean()), 2)
        ds_bucket[m] = {
            "subject_means": s_means,
            "subject_finals": s_finals,
            "subject_kappas": s_kappas,
            "mean": mean_v,
            "sd": round(float(vals.std()), 2),
            "final_mean": fin_v,
            "final_sd": round(float(fvals.std()), 2),
            "peek_inflation": round(max(0.0, mean_v - fin_v), 2),
            "max": round(float(vals.max()), 2),
            "n_subjects": len(s_ids),
            "curve": mean_curve,
        }

    # 4. Write aggregated SPD matrix dataset x model summaries
    for (ds_k, var), seed_map in sorted(spd_grouped.items()):
        ds_bucket = res.setdefault(ds_k, {})
        seeds = sorted(seed_map.keys())
        accs = [round(float(seed_map[s]["accuracy"]), 2) for s in seeds]
        fin_accs = [round(float(seed_map[s].get("final_accuracy", seed_map[s]["accuracy"])), 2) for s in seeds]
        arr = np.array(accs, dtype=np.float64)
        farr = np.array(fin_accs, dtype=np.float64)
        curves = [seed_map[s].get("curve", [])[:6] for s in seeds if len(seed_map[s].get("curve", [])) >= 6]
        mean_curve = (
            [round(float(x), 2) for x in np.mean(np.array(curves, dtype=np.float64), axis=0)]
            if curves
            else [round(float(arr.mean()), 2)] * 6
        )
        walls = [float(seed_map[s].get("wall_s", 1.0)) for s in seeds]
        ds_bucket[var] = {
            "dataset": ds_k,
            "variant": var,
            "n_seeds": len(seeds),
            "seeds": seeds,
            "seed_accuracies": accs,
            "accuracy": round(float(arr.mean()), 2),
            "sd": round(float(arr.std()), 2),
            "max_accuracy": round(float(arr.max()), 2),
            "final_accuracy": round(float(farr.mean()), 2),
            "curve": mean_curve,
            "wall_s": round(float(np.mean(walls)), 2),
        }

    # 4b. Incrementally merge s56_*.json and full_*.json into existing EEG & SPD buckets
    full_ch_replace_map = {
        "full_mamem2.json": ["mamem2"],
        "full_weibo.json": ["weibo2014"],
        "full_hinss.json": ["hinss2021"],
        "full_bciern.json": ["bci_ern"],
        "full_zhou.json": ["zhou2020"],
        "full_cho_lee.json": ["cho2017", "lee2019_mi"],
        "full_physio_stieger.json": ["physionet_mi", "stieger2021"],
        "full_schirr_p1.json": ["schirrmeister2017"],
    }
    pre_files = sorted(glob.glob(str(SHARDS_DIR / "s56_*.json")))
    if (SHARDS_DIR / "full_repair.json").exists():
        pre_files.append(str(SHARDS_DIR / "full_repair.json"))
    post_files = sorted(
        sp
        for sp in glob.glob(str(SHARDS_DIR / "full_*.json"))
        if Path(sp).name not in {"full_schirr.json", "full_repair.json"}
    )

    def _apply_shard_files(file_list: list[str]) -> None:
        for sp in file_list:
            payload = json.loads(Path(sp).read_text())
            for r in payload.get("eeg_records", []):
                ds_k = DS_NAME_TO_KEY.get(r["dataset"], r["dataset"].lower())
                m = r["model"]
                s_str = str(int(r["subject"]))
                ds_bucket = eeg_ds.setdefault(ds_k, {})
                entry = ds_bucket.setdefault(
                    m,
                    {
                        "subject_means": {},
                        "subject_finals": {},
                        "subject_kappas": {},
                        "curve": r.get("curve", []),
                    },
                )
                entry.setdefault("subject_means", {})[s_str] = round(float(r["balanced_accuracy"]), 2)
                entry.setdefault("subject_finals", {})[s_str] = round(
                    float(r.get("final_balanced_accuracy", r["balanced_accuracy"])), 2
                )
                entry.setdefault("subject_kappas", {})[s_str] = round(float(r.get("kappa", 0.0)), 4)
                s_keys = sorted(entry["subject_means"].keys(), key=lambda x: int(x))
                entry["subject_means"] = {k: entry["subject_means"][k] for k in s_keys}
                entry["subject_finals"] = {
                    k: entry["subject_finals"].get(k, entry["subject_means"][k]) for k in s_keys
                }
                entry["subject_kappas"] = {
                    k: entry["subject_kappas"].get(k, 0.0) for k in s_keys
                }
                vals = np.array(list(entry["subject_means"].values()), dtype=np.float64)
                fvals = np.array(list(entry["subject_finals"].values()), dtype=np.float64)
                mean_v = round(float(vals.mean()), 2)
                fin_v = round(float(fvals.mean()), 2)
                entry["mean"] = mean_v
                entry["sd"] = round(float(vals.std()), 2)
                entry["final_mean"] = fin_v
                entry["final_sd"] = round(float(fvals.std()), 2)
                entry["peek_inflation"] = round(max(0.0, mean_v - fin_v), 2)
                entry["max"] = round(float(vals.max()), 2)
                entry["n_subjects"] = len(s_keys)

            for r in payload.get("spd_records", []):
                ds_k = r["dataset"].lower()
                var = r["variant"]
                seed = int(r["seed"])
                ds_bucket = res.setdefault(ds_k, {})
                entry = ds_bucket.setdefault(
                    var,
                    {
                        "dataset": ds_k,
                        "variant": var,
                        "seeds": [],
                        "seed_accuracies": [],
                        "curve": r.get("curve", [])[:6],
                        "wall_s": round(float(r.get("wall_s", 1.0)), 2),
                    },
                )
                seed_to_acc = dict(
                    zip(entry.get("seeds", []), entry.get("seed_accuracies", []))
                )
                seed_to_acc[seed] = round(float(r["accuracy"]), 2)
                if {1024, 1025, 1026, 1027, 1028} <= set(seed_to_acc.keys()):
                    seed_to_acc = {k: v for k, v in seed_to_acc.items() if k in {1024, 1025, 1026, 1027, 1028}}
                seeds_sorted = sorted(seed_to_acc.keys())
                accs_sorted = [seed_to_acc[s] for s in seeds_sorted]
                arr = np.array(accs_sorted, dtype=np.float64)
                entry["seeds"] = seeds_sorted
                entry["seed_accuracies"] = accs_sorted
                entry["n_seeds"] = len(seeds_sorted)
                entry["accuracy"] = round(float(arr.mean()), 2)
                entry["sd"] = round(float(arr.std()), 2)
                entry["max_accuracy"] = round(float(arr.max()), 2)
                if "final_accuracy" not in entry:
                    entry["final_accuracy"] = entry["accuracy"]

    _apply_shard_files(pre_files)
    for shard_fn, ds_keys_clear in full_ch_replace_map.items():
        if (SHARDS_DIR / shard_fn).exists():
            for dk in ds_keys_clear:
                eeg_ds[dk] = {}
    _apply_shard_files(post_files)

    # 5. Recompute total run counts across all 29 datasets
    total_eeg_runs = sum(
        int(m_rec.get("n_subjects", 1))
        for ds_dict in eeg_ds.values()
        for m_rec in ds_dict.values()
    )
    total_spd_runs = sum(
        int(m_rec.get("n_seeds", 1))
        for ds_k in ("radar", "hdm05", "fpha", "afew")
        for m_rec in res.get(ds_k, {}).values()
    )
    res.setdefault("heavy_campaign_summary", {})
    res["heavy_campaign_summary"]["n_eeg_datasets"] = len(eeg_ds)
    res["heavy_campaign_summary"]["n_spd_datasets"] = 4
    res["heavy_campaign_summary"]["n_total_datasets"] = len(eeg_ds) + 4
    res["heavy_campaign_summary"]["n_eeg_runs"] = total_eeg_runs
    res["heavy_campaign_summary"]["n_spd_runs"] = total_spd_runs
    res["heavy_campaign_summary"]["n_total_runs"] = total_eeg_runs + total_spd_runs

    RESULTS_PATH.write_text(json.dumps(res, indent=2) + "\n")
    print(
        f"Merged voyager_spdbench_results.json: {len(eeg_ds)} EEG datasets ({total_eeg_runs} runs) + "
        f"4 SPD datasets ({total_spd_runs} runs) = {total_eeg_runs + total_spd_runs} total runs."
    )
    for ds_k in sorted(eeg_ds.keys()):
        print(f"  EEG {ds_k:18s}: {len(eeg_ds[ds_k]):2d} models")
    for ds_k in ("radar", "hdm05", "fpha", "afew"):
        print(f"  SPD {ds_k:18s}: {len(res.get(ds_k, {})):2d} models")


if __name__ == "__main__":
    merge_gen_shards()
