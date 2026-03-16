#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "analysis" / "historical_driver_features.csv"
SPLIT_PATH = ROOT / "analysis" / "splits" / "race_id_splits.json"
OUTPUT_DIR = ROOT / "analysis" / "models" / "physics_simulator"
PARAMS_PATH = OUTPUT_DIR / "best_params.json"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


def load_data():
    df = pd.read_csv(DATA_PATH)
    splits = json.loads(SPLIT_PATH.read_text())
    return df, splits


def add_stint_sufficient_statistics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for compound in COMPOUNDS:
        out[f"{compound.lower()}_age_sum"] = 0.0
        out[f"{compound.lower()}_age_sq_sum"] = 0.0

    for stint_idx in (1, 2, 3):
        comp_col = f"stint_{stint_idx}_compound"
        laps_col = f"stint_{stint_idx}_laps"
        laps = out[laps_col].fillna(0).astype(float)
        age_sum = laps * (laps + 1.0) / 2.0
        age_sq_sum = laps * (laps + 1.0) * (2.0 * laps + 1.0) / 6.0
        for compound in COMPOUNDS:
            mask = out[comp_col] == compound
            out.loc[mask, f"{compound.lower()}_age_sum"] += age_sum[mask]
            out.loc[mask, f"{compound.lower()}_age_sq_sum"] += age_sq_sum[mask]

    return out


def predict_total_time(frame: pd.DataFrame, params: dict) -> np.ndarray:
    pred = frame["base_lap_time"].to_numpy(dtype=float) * frame["total_laps"].to_numpy(dtype=float)
    pred += frame["pit_lane_time"].to_numpy(dtype=float) * frame["pit_stop_count"].to_numpy(dtype=float)

    temp = frame["track_temp"].to_numpy(dtype=float)
    for compound in COMPOUNDS:
        prefix = compound.lower()
        laps = frame[f"{prefix}_laps"].to_numpy(dtype=float)
        age_sum = frame[f"{prefix}_age_sum"].to_numpy(dtype=float)
        age_sq_sum = frame[f"{prefix}_age_sq_sum"].to_numpy(dtype=float)

        pred += params[f"{prefix}_offset"] * laps
        pred += params[f"{prefix}_linear"] * age_sum
        pred += params[f"{prefix}_quadratic"] * age_sq_sum
        pred += params[f"{prefix}_temp_linear"] * temp * age_sum

    return pred


def rank_metrics(frame: pd.DataFrame, pred_score: np.ndarray) -> dict:
    scored = frame[["race_id", "driver_id", "finish_rank"]].copy()
    scored["pred_score"] = pred_score

    exact = 0
    top3 = 0
    top5 = 0
    kendalls = []
    spearmans = []

    for _, race in scored.groupby("race_id", sort=False):
        predicted = race.sort_values(["pred_score", "driver_id"], ascending=[True, True])
        actual = race.sort_values(["finish_rank", "driver_id"], ascending=[True, True])
        pred_ids = predicted["driver_id"].tolist()
        actual_ids = actual["driver_id"].tolist()
        exact += int(pred_ids == actual_ids)
        top3 += int(set(pred_ids[:3]) == set(actual_ids[:3]))
        top5 += int(set(pred_ids[:5]) == set(actual_ids[:5]))

        merged = actual[["driver_id", "finish_rank"]].merge(
            predicted[["driver_id"]].assign(pred_rank=np.arange(1, len(predicted) + 1)),
            on="driver_id",
            how="inner",
        )
        kendalls.append(float(kendalltau(merged["finish_rank"], merged["pred_rank"]).statistic))
        spearmans.append(float(spearmanr(merged["finish_rank"], merged["pred_rank"]).statistic))

    race_count = scored["race_id"].nunique()
    return {
        "races": int(race_count),
        "exact_order_accuracy": exact / race_count,
        "top3_set_accuracy": top3 / race_count,
        "top5_set_accuracy": top5 / race_count,
        "mean_kendall_tau": float(np.mean(kendalls)),
        "mean_spearman_rho": float(np.mean(spearmans)),
    }


def objective(frame: pd.DataFrame, params: dict) -> float:
    metrics = rank_metrics(frame, predict_total_time(frame, params))
    return metrics["mean_kendall_tau"]


def sample_params(rng: np.random.Generator) -> dict:
    # Pace ordering prior: SOFT fastest, HARD slowest.
    soft_offset = rng.uniform(-2.5, -0.2)
    medium_offset = rng.uniform(-1.0, 1.0)
    hard_offset = rng.uniform(0.2, 2.5)

    params = {
        "soft_offset": soft_offset,
        "medium_offset": medium_offset,
        "hard_offset": hard_offset,
    }

    for prefix, lin_lo, lin_hi, quad_lo, quad_hi, temp_lo, temp_hi in (
        ("soft", 0.02, 0.20, -0.001, 0.008, 0.0000, 0.0040),
        ("medium", 0.01, 0.12, -0.001, 0.006, 0.0000, 0.0030),
        ("hard", 0.00, 0.08, -0.001, 0.004, -0.0005, 0.0025),
    ):
        params[f"{prefix}_linear"] = rng.uniform(lin_lo, lin_hi)
        params[f"{prefix}_quadratic"] = rng.uniform(quad_lo, quad_hi)
        params[f"{prefix}_temp_linear"] = rng.uniform(temp_lo, temp_hi)
    return params


def calibrate(train_frame: pd.DataFrame, validation_frame: pd.DataFrame, iterations: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    best = None

    for iteration in range(iterations):
        params = sample_params(rng)
        train_score = objective(train_frame, params)
        val_metrics = rank_metrics(validation_frame, predict_total_time(validation_frame, params))
        record = {
            "iteration": iteration,
            "params": params,
            "train_kendall": train_score,
            "validation_metrics": val_metrics,
        }
        if best is None or val_metrics["mean_kendall_tau"] > best["validation_metrics"]["mean_kendall_tau"]:
            best = record
            print(
                f"iter={iteration} "
                f"train_kendall={train_score:.4f} "
                f"val_kendall={val_metrics['mean_kendall_tau']:.4f} "
                f"val_spearman={val_metrics['mean_spearman_rho']:.4f}"
            , flush=True)
    return best


def main():
    df, splits = load_data()
    df = add_stint_sufficient_statistics(df)

    train = df[df["race_id"].isin(splits["train"])].copy()
    validation = df[df["race_id"].isin(splits["validation"])].copy()
    test = df[df["race_id"].isin(splits["test"])].copy()

    # Use a race sample for faster calibration, then evaluate on full val/test.
    sampled_train_races = train["race_id"].drop_duplicates().sample(n=200, random_state=42)
    sampled_validation_races = validation["race_id"].drop_duplicates().sample(n=100, random_state=42)
    train_sample = train[train["race_id"].isin(sampled_train_races)].copy()
    validation_sample = validation[validation["race_id"].isin(sampled_validation_races)].copy()

    best = calibrate(train_sample, validation_sample)
    best_params = best["params"]

    full_metrics = {
        "validation": rank_metrics(validation, predict_total_time(validation, best_params)),
        "test": rank_metrics(test, predict_total_time(test, best_params)),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(best, indent=2))
    METRICS_PATH.write_text(json.dumps(full_metrics, indent=2))

    print("\nBest parameters:")
    print(json.dumps(best_params, indent=2))
    print("\nFull evaluation:")
    print(json.dumps(full_metrics, indent=2))


if __name__ == "__main__":
    main()
