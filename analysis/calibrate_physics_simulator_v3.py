#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "analysis" / "historical_driver_features.csv"
SPLIT_PATH = ROOT / "analysis" / "splits" / "race_id_splits.json"
OUTPUT_DIR = ROOT / "analysis" / "models" / "physics_simulator_v3"
PARAMS_PATH = OUTPUT_DIR / "best_params.json"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
TRACKS = ("Bahrain", "COTA", "Monaco", "Monza", "Silverstone", "Spa", "Suzuka")


def load_data():
    df = pd.read_csv(DATA_PATH)
    splits = json.loads(SPLIT_PATH.read_text())
    return df, splits


def stint_contribution(laps: np.ndarray, threshold: int):
    ages = laps.astype(int)
    fresh_laps = np.minimum(ages, threshold).astype(float)
    worn_laps = np.maximum(ages - threshold, 0).astype(float)

    fresh_age_sum = fresh_laps * (fresh_laps + 1.0) / 2.0
    excess_age_sum = worn_laps * (worn_laps + 1.0) / 2.0
    excess_age_sq_sum = worn_laps * (worn_laps + 1.0) * (2.0 * worn_laps + 1.0) / 6.0

    return fresh_age_sum, excess_age_sum, excess_age_sq_sum


def add_piecewise_statistics(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    out = df.copy()
    for compound in COMPOUNDS:
        prefix = compound.lower()
        out[f"{prefix}_fresh_age_sum"] = 0.0
        out[f"{prefix}_excess_age_sum"] = 0.0
        out[f"{prefix}_excess_age_sq_sum"] = 0.0

    for stint_idx in (1, 2, 3):
        comp_col = f"stint_{stint_idx}_compound"
        laps_col = f"stint_{stint_idx}_laps"
        laps = out[laps_col].fillna(0).astype(float)
        for compound in COMPOUNDS:
            prefix = compound.lower()
            mask = out[comp_col] == compound
            if not mask.any():
                continue
            fresh_age_sum, excess_age_sum, excess_age_sq_sum = stint_contribution(
                laps[mask].to_numpy(dtype=float), thresholds[compound]
            )
            out.loc[mask, f"{prefix}_fresh_age_sum"] += fresh_age_sum
            out.loc[mask, f"{prefix}_excess_age_sum"] += excess_age_sum
            out.loc[mask, f"{prefix}_excess_age_sq_sum"] += excess_age_sq_sum

    return out


def predict_total_time(frame: pd.DataFrame, params: dict, thresholds: dict) -> np.ndarray:
    pred = frame["base_lap_time"].to_numpy(dtype=float) * frame["total_laps"].to_numpy(dtype=float)
    pred += frame["pit_lane_time"].to_numpy(dtype=float) * frame["pit_stop_count"].to_numpy(dtype=float)
    temp = frame["track_temp"].to_numpy(dtype=float)
    track_values = frame["track"].to_numpy(dtype=object)

    for compound in COMPOUNDS:
        prefix = compound.lower()
        laps = frame[f"{prefix}_laps"].to_numpy(dtype=float)
        fresh_age_sum = frame[f"{prefix}_fresh_age_sum"].to_numpy(dtype=float)
        excess_age_sum = frame[f"{prefix}_excess_age_sum"].to_numpy(dtype=float)
        excess_age_sq_sum = frame[f"{prefix}_excess_age_sq_sum"].to_numpy(dtype=float)

        wear_multiplier = np.ones(len(frame), dtype=float)
        for track in TRACKS:
            wear_multiplier[track_values == track] = params["track_wear_multiplier"][track]

        pred += params[f"{prefix}_offset"] * laps
        pred += params[f"{prefix}_fresh_linear"] * fresh_age_sum
        pred += wear_multiplier * params[f"{prefix}_wear_linear"] * excess_age_sum
        pred += wear_multiplier * params[f"{prefix}_wear_quadratic"] * excess_age_sq_sum
        pred += wear_multiplier * params[f"{prefix}_temp_wear"] * temp * excess_age_sum

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


def sample_params(rng: np.random.Generator) -> tuple[dict, dict]:
    thresholds = {
        "SOFT": int(rng.integers(4, 10)),
        "MEDIUM": int(rng.integers(8, 17)),
        "HARD": int(rng.integers(12, 25)),
    }

    params = {
        "soft_offset": rng.uniform(-2.0, -0.2),
        "medium_offset": rng.uniform(-0.8, 0.8),
        "hard_offset": rng.uniform(0.0, 1.5),
        "track_wear_multiplier": {
            track: rng.uniform(0.8, 1.25) for track in TRACKS
        },
    }

    for prefix, fresh_lo, fresh_hi, wear_lo, wear_hi, quad_lo, quad_hi, temp_lo, temp_hi in (
        ("soft", 0.0, 0.05, 0.08, 0.30, 0.001, 0.010, 0.0005, 0.0060),
        ("medium", 0.0, 0.04, 0.04, 0.18, 0.0005, 0.007, 0.0002, 0.0040),
        ("hard", -0.01, 0.03, 0.02, 0.12, 0.0000, 0.005, 0.0000, 0.0030),
    ):
        params[f"{prefix}_fresh_linear"] = rng.uniform(fresh_lo, fresh_hi)
        params[f"{prefix}_wear_linear"] = rng.uniform(wear_lo, wear_hi)
        params[f"{prefix}_wear_quadratic"] = rng.uniform(quad_lo, quad_hi)
        params[f"{prefix}_temp_wear"] = rng.uniform(temp_lo, temp_hi)

    return params, thresholds


def calibrate(train_base: pd.DataFrame, validation_base: pd.DataFrame, iterations: int = 30, seed: int = 42):
    rng = np.random.default_rng(seed)
    best = None

    for iteration in range(iterations):
        params, thresholds = sample_params(rng)
        train_frame = add_piecewise_statistics(train_base, thresholds)
        validation_frame = add_piecewise_statistics(validation_base, thresholds)

        train_metrics = rank_metrics(train_frame, predict_total_time(train_frame, params, thresholds))
        val_metrics = rank_metrics(validation_frame, predict_total_time(validation_frame, params, thresholds))
        record = {
            "iteration": iteration,
            "params": params,
            "thresholds": thresholds,
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics,
        }
        if best is None or val_metrics["mean_kendall_tau"] > best["validation_metrics"]["mean_kendall_tau"]:
            best = record
            print(
                f"iter={iteration} "
                f"train_kendall={train_metrics['mean_kendall_tau']:.4f} "
                f"val_kendall={val_metrics['mean_kendall_tau']:.4f} "
                f"val_spearman={val_metrics['mean_spearman_rho']:.4f}",
                flush=True,
            )
    return best


def main():
    df, splits = load_data()
    train = df[df["race_id"].isin(splits["train"])].copy()
    validation = df[df["race_id"].isin(splits["validation"])].copy()
    test = df[df["race_id"].isin(splits["test"])].copy()

    sampled_train_races = train["race_id"].drop_duplicates().sample(n=200, random_state=42)
    sampled_validation_races = validation["race_id"].drop_duplicates().sample(n=100, random_state=42)
    train_sample = train[train["race_id"].isin(sampled_train_races)].copy()
    validation_sample = validation[validation["race_id"].isin(sampled_validation_races)].copy()

    best = calibrate(train_sample, validation_sample)
    best_params = best["params"]
    best_thresholds = best["thresholds"]

    validation_eval = add_piecewise_statistics(validation, best_thresholds)
    test_eval = add_piecewise_statistics(test, best_thresholds)

    full_metrics = {
        "validation": rank_metrics(validation_eval, predict_total_time(validation_eval, best_params, best_thresholds)),
        "test": rank_metrics(test_eval, predict_total_time(test_eval, best_params, best_thresholds)),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(best, indent=2))
    METRICS_PATH.write_text(json.dumps(full_metrics, indent=2))

    print("\nBest thresholds:")
    print(json.dumps(best_thresholds, indent=2))
    print("\nBest track wear multipliers:")
    print(json.dumps(best_params["track_wear_multiplier"], indent=2))
    print("\nFull evaluation:")
    print(json.dumps(full_metrics, indent=2))


if __name__ == "__main__":
    main()
