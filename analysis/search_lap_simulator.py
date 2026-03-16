#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "analysis" / "historical_driver_features.csv"
SPLIT_PATH = ROOT / "analysis" / "splits" / "race_id_splits.json"
OUTPUT_DIR = ROOT / "analysis" / "lap_sim_search"
RESULTS_PATH = OUTPUT_DIR / "results.json"

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


def load_data():
    df = pd.read_csv(DATA_PATH)
    splits = json.loads(SPLIT_PATH.read_text())
    return df, splits


def add_regimes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lap_band"] = pd.cut(
        out["total_laps"],
        bins=[24, 35, 45, 55, 70],
        labels=["short", "mid_short", "mid", "long"],
        include_lowest=True,
    ).astype(str)
    out["temp_band"] = pd.cut(
        out["track_temp"],
        bins=[17, 24, 30, 36, 42],
        labels=["cool", "mild", "warm", "hot"],
        include_lowest=True,
    ).astype(str)
    out["first_stop_frac"] = out["first_stop_lap"].fillna(0.0) / out["total_laps"]
    out["second_stop_frac"] = out["second_stop_lap"].fillna(0.0) / out["total_laps"]
    return out


def s1(n):
    return n * (n + 1.0) / 2.0


def s2(n):
    return n * (n + 1.0) * (2.0 * n + 1.0) / 6.0


def stint_stats(laps: np.ndarray, fresh_threshold: int, cliff_threshold: int):
    ages = laps.astype(float)
    fresh = np.minimum(ages, fresh_threshold)
    worn = np.maximum(ages - fresh_threshold, 0.0)
    cliff = np.maximum(ages - cliff_threshold, 0.0)
    return {
        "fresh_age_sum": s1(fresh),
        "worn_age_sum": s1(worn),
        "worn_age_sq_sum": s2(worn),
        "cliff_age_sum": s1(cliff),
    }


def add_lap_terms(df: pd.DataFrame, thresholds: dict, cliff_thresholds: dict) -> pd.DataFrame:
    out = df.copy()
    for compound in COMPOUNDS:
        prefix = compound.lower()
        out[f"{prefix}_fresh_age_sum"] = 0.0
        out[f"{prefix}_worn_age_sum"] = 0.0
        out[f"{prefix}_worn_age_sq_sum"] = 0.0
        out[f"{prefix}_cliff_age_sum"] = 0.0

    for stint_idx in (1, 2, 3):
        comp_col = f"stint_{stint_idx}_compound"
        laps_col = f"stint_{stint_idx}_laps"
        laps = out[laps_col].fillna(0.0).astype(float)
        for compound in COMPOUNDS:
            mask = out[comp_col] == compound
            if not mask.any():
                continue
            stats = stint_stats(
                laps[mask].to_numpy(),
                thresholds[compound],
                cliff_thresholds[compound],
            )
            prefix = compound.lower()
            for k, v in stats.items():
                out.loc[mask, f"{prefix}_{k}"] += v
    return out


def score_frame(frame: pd.DataFrame, params: dict) -> np.ndarray:
    total_laps = frame["total_laps"].to_numpy(dtype=float)
    track_temp = frame["track_temp"].to_numpy(dtype=float)
    pred = frame["base_lap_time"].to_numpy(dtype=float) * total_laps
    pred += frame["pit_lane_time"].to_numpy(dtype=float) * frame["pit_stop_count"].to_numpy(dtype=float)
    pred += params["first_stop_frac"] * frame["first_stop_frac"].to_numpy(dtype=float)
    pred += params["second_stop_frac"] * frame["second_stop_frac"].to_numpy(dtype=float)
    pred += params["stint_2_laps"] * frame["stint_2_laps"].to_numpy(dtype=float)
    pred += params["stint_3_laps"] * frame["stint_3_laps"].to_numpy(dtype=float)

    lap_short = (frame["lap_band"] == "short").to_numpy(dtype=float)
    lap_mid_short = (frame["lap_band"] == "mid_short").to_numpy(dtype=float)
    lap_mid = (frame["lap_band"] == "mid").to_numpy(dtype=float)
    lap_long = (frame["lap_band"] == "long").to_numpy(dtype=float)
    temp_cool = (frame["temp_band"] == "cool").to_numpy(dtype=float)
    temp_mild = (frame["temp_band"] == "mild").to_numpy(dtype=float)
    temp_warm = (frame["temp_band"] == "warm").to_numpy(dtype=float)
    temp_hot = (frame["temp_band"] == "hot").to_numpy(dtype=float)

    for key, mask in {
        "lap_short": lap_short,
        "lap_mid_short": lap_mid_short,
        "lap_mid": lap_mid,
        "lap_long": lap_long,
        "temp_cool": temp_cool,
        "temp_mild": temp_mild,
        "temp_warm": temp_warm,
        "temp_hot": temp_hot,
    }.items():
        pred += params[key] * mask

    for compound in COMPOUNDS:
        prefix = compound.lower()
        laps = frame[f"{prefix}_laps"].to_numpy(dtype=float)
        fresh_age_sum = frame[f"{prefix}_fresh_age_sum"].to_numpy(dtype=float)
        worn_age_sum = frame[f"{prefix}_worn_age_sum"].to_numpy(dtype=float)
        worn_age_sq_sum = frame[f"{prefix}_worn_age_sq_sum"].to_numpy(dtype=float)
        cliff_age_sum = frame[f"{prefix}_cliff_age_sum"].to_numpy(dtype=float)
        frac = laps / total_laps

        pred += params[f"{prefix}_offset"] * laps
        pred += params[f"{prefix}_fresh_linear"] * fresh_age_sum
        pred += params[f"{prefix}_wear_linear"] * worn_age_sum
        pred += params[f"{prefix}_wear_quadratic"] * worn_age_sq_sum
        pred += params[f"{prefix}_temp_linear"] * track_temp * worn_age_sum
        pred += params[f"{prefix}_cliff_penalty"] * cliff_age_sum
        pred += params[f"{prefix}_fraction_bias"] * frac
        pred += params[f"{prefix}_lap_short"] * lap_short * laps
        pred += params[f"{prefix}_lap_long"] * lap_long * laps
        pred += params[f"{prefix}_temp_cool"] * temp_cool * laps
        pred += params[f"{prefix}_temp_hot"] * temp_hot * laps
    return pred


def evaluate(frame: pd.DataFrame, pred_score: np.ndarray) -> dict:
    scored = frame[["race_id", "driver_id", "finish_rank"]].copy()
    scored["pred_score"] = pred_score
    exact = 0
    kendalls = []
    spearmans = []
    for _, race in scored.groupby("race_id", sort=False):
        predicted = race.sort_values(["pred_score", "driver_id"], ascending=[True, True])
        actual = race.sort_values(["finish_rank", "driver_id"], ascending=[True, True])
        pred_ids = predicted["driver_id"].tolist()
        actual_ids = actual["driver_id"].tolist()
        exact += int(pred_ids == actual_ids)
        merged = actual[["driver_id", "finish_rank"]].merge(
            predicted[["driver_id"]].assign(pred_rank=np.arange(1, len(predicted) + 1)),
            on="driver_id",
            how="inner",
        )
        kendalls.append(float(kendalltau(merged["finish_rank"], merged["pred_rank"]).statistic))
        spearmans.append(float(spearmanr(merged["finish_rank"], merged["pred_rank"]).statistic))
    races = frame["race_id"].nunique()
    return {
        "races": int(races),
        "exact_order_accuracy": exact / races,
        "mean_kendall_tau": float(np.mean(kendalls)),
        "mean_spearman_rho": float(np.mean(spearmans)),
    }


def sample_candidate(rng: np.random.Generator):
    thresholds = {
        "SOFT": int(rng.integers(4, 10)),
        "MEDIUM": int(rng.integers(8, 18)),
        "HARD": int(rng.integers(12, 26)),
    }
    cliff_thresholds = {
        "SOFT": int(rng.integers(thresholds["SOFT"] + 1, 16)),
        "MEDIUM": int(rng.integers(thresholds["MEDIUM"] + 1, 27)),
        "HARD": int(rng.integers(thresholds["HARD"] + 1, 37)),
    }
    params = {
        "first_stop_frac": rng.uniform(-3.0, 3.0),
        "second_stop_frac": rng.uniform(-1.5, 1.5),
        "stint_2_laps": rng.uniform(-0.3, 0.3),
        "stint_3_laps": rng.uniform(-0.3, 0.3),
        "lap_short": rng.uniform(-2.0, 2.0),
        "lap_mid_short": rng.uniform(-2.0, 2.0),
        "lap_mid": rng.uniform(-2.0, 2.0),
        "lap_long": rng.uniform(-2.0, 2.0),
        "temp_cool": rng.uniform(-1.0, 1.0),
        "temp_mild": rng.uniform(-1.0, 1.0),
        "temp_warm": rng.uniform(-1.0, 1.0),
        "temp_hot": rng.uniform(-1.0, 1.0),
    }
    ranges = {
        "soft": (-2.5, -0.2, 0.0, 0.05, 0.05, 0.35, 0.0005, 0.012, 0.0002, 0.007, 0.03, 0.40, -2.0, 2.0),
        "medium": (-1.2, 0.8, 0.0, 0.04, 0.02, 0.20, 0.0002, 0.008, 0.0001, 0.005, 0.01, 0.25, -2.0, 2.0),
        "hard": (0.0, 1.8, -0.01, 0.03, 0.01, 0.14, 0.0000, 0.006, 0.0000, 0.004, 0.00, 0.15, -2.0, 2.0),
    }
    for prefix, vals in ranges.items():
        offset_lo, offset_hi, fresh_lo, fresh_hi, wear_lo, wear_hi, quad_lo, quad_hi, temp_lo, temp_hi, cliff_lo, cliff_hi, frac_lo, frac_hi = vals
        params[f"{prefix}_offset"] = rng.uniform(offset_lo, offset_hi)
        params[f"{prefix}_fresh_linear"] = rng.uniform(fresh_lo, fresh_hi)
        params[f"{prefix}_wear_linear"] = rng.uniform(wear_lo, wear_hi)
        params[f"{prefix}_wear_quadratic"] = rng.uniform(quad_lo, quad_hi)
        params[f"{prefix}_temp_linear"] = rng.uniform(temp_lo, temp_hi)
        params[f"{prefix}_cliff_penalty"] = rng.uniform(cliff_lo, cliff_hi)
        params[f"{prefix}_fraction_bias"] = rng.uniform(frac_lo, frac_hi)
        params[f"{prefix}_lap_short"] = rng.uniform(-0.10, 0.10)
        params[f"{prefix}_lap_long"] = rng.uniform(-0.10, 0.10)
        params[f"{prefix}_temp_cool"] = rng.uniform(-0.10, 0.10)
        params[f"{prefix}_temp_hot"] = rng.uniform(-0.10, 0.10)
    return params, thresholds, cliff_thresholds


def search(train: pd.DataFrame, validation: pd.DataFrame, iterations: int = 120, seed: int = 42):
    rng = np.random.default_rng(seed)
    best = []
    for i in range(iterations):
        params, thresholds, cliff_thresholds = sample_candidate(rng)
        train_terms = add_lap_terms(train, thresholds, cliff_thresholds)
        val_terms = add_lap_terms(validation, thresholds, cliff_thresholds)
        train_metrics = evaluate(train_terms, score_frame(train_terms, params))
        val_metrics = evaluate(val_terms, score_frame(val_terms, params))
        rec = {
            "iteration": i,
            "params": params,
            "thresholds": thresholds,
            "cliff_thresholds": cliff_thresholds,
            "train": train_metrics,
            "validation": val_metrics,
        }
        best.append(rec)
        best.sort(key=lambda r: (r["validation"]["exact_order_accuracy"], r["validation"]["mean_kendall_tau"]), reverse=True)
        best = best[:15]
        if best[0]["iteration"] == i:
            print(json.dumps({
                "iteration": i,
                "val_exact": val_metrics["exact_order_accuracy"],
                "val_kendall": val_metrics["mean_kendall_tau"],
                "val_spearman": val_metrics["mean_spearman_rho"],
            }), flush=True)
    return best


def main():
    df, splits = load_data()
    df = add_regimes(df)
    train = df[df["race_id"].isin(splits["train"])].copy()
    validation = df[df["race_id"].isin(splits["validation"])].copy()
    test = df[df["race_id"].isin(splits["test"])].copy()

    train_races = train["race_id"].drop_duplicates().sample(n=250, random_state=42)
    val_races = validation["race_id"].drop_duplicates().sample(n=120, random_state=42)
    train_sample = train[train["race_id"].isin(train_races)].copy()
    validation_sample = validation[validation["race_id"].isin(val_races)].copy()

    best = search(train_sample, validation_sample)
    top_results = []
    for rec in best[:10]:
        test_terms = add_lap_terms(test, rec["thresholds"], rec["cliff_thresholds"])
        test_metrics = evaluate(test_terms, score_frame(test_terms, rec["params"]))
        top_results.append(
            {
                "iteration": rec["iteration"],
                "thresholds": rec["thresholds"],
                "cliff_thresholds": rec["cliff_thresholds"],
                "train": rec["train"],
                "validation": rec["validation"],
                "test": test_metrics,
                "params": rec["params"],
            }
        )
    top_results.sort(key=lambda r: (r["test"]["exact_order_accuracy"], r["test"]["mean_kendall_tau"]), reverse=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(top_results, indent=2))
    print("\nTop results:")
    print(json.dumps(top_results[:5], indent=2))


if __name__ == "__main__":
    main()
