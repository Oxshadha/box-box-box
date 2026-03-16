#!/usr/bin/env python3
import json
from pathlib import Path

import calibrate_physics_simulator_v4 as base_v4
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "analysis" / "historical_driver_features.csv"
SPLIT_PATH = ROOT / "analysis" / "splits" / "race_id_splits.json"
OUTPUT_DIR = ROOT / "analysis" / "models" / "physics_simulator_v4_front"
PARAMS_PATH = OUTPUT_DIR / "best_params.json"
METRICS_PATH = OUTPUT_DIR / "metrics.json"


def selection_score(metrics):
    return (
        0.45 * metrics["top3_set_accuracy"]
        + 0.35 * metrics["top5_set_accuracy"]
        + 0.15 * metrics["exact_order_accuracy"]
        + 0.05 * metrics["mean_kendall_tau"]
    )


def calibrate(train_base: pd.DataFrame, validation_base: pd.DataFrame, iterations: int = 35, seed: int = 42):
    rng = base_v4.np.random.default_rng(seed)
    best = None

    for iteration in range(iterations):
        params, thresholds, cliff_thresholds = base_v4.sample_params(rng)
        train_frame = base_v4.add_piecewise_statistics(train_base, thresholds, cliff_thresholds)
        validation_frame = base_v4.add_piecewise_statistics(validation_base, thresholds, cliff_thresholds)

        train_metrics = base_v4.rank_metrics(train_frame, base_v4.predict_total_time(train_frame, params))
        val_metrics = base_v4.rank_metrics(validation_frame, base_v4.predict_total_time(validation_frame, params))
        record = {
            "iteration": iteration,
            "params": params,
            "thresholds": thresholds,
            "cliff_thresholds": cliff_thresholds,
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics,
            "selection_score": selection_score(val_metrics),
        }
        if best is None or record["selection_score"] > best["selection_score"]:
            best = record
            print(
                f"iter={iteration} "
                f"val_exact={val_metrics['exact_order_accuracy']:.4f} "
                f"val_top3={val_metrics['top3_set_accuracy']:.4f} "
                f"val_top5={val_metrics['top5_set_accuracy']:.4f} "
                f"val_kendall={val_metrics['mean_kendall_tau']:.4f}",
                flush=True,
            )
    return best


def main():
    df = pd.read_csv(DATA_PATH)
    splits = json.loads(SPLIT_PATH.read_text())
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
    best_cliff_thresholds = best["cliff_thresholds"]

    validation_eval = base_v4.add_piecewise_statistics(validation, best_thresholds, best_cliff_thresholds)
    test_eval = base_v4.add_piecewise_statistics(test, best_thresholds, best_cliff_thresholds)

    full_metrics = {
        "validation": base_v4.rank_metrics(validation_eval, base_v4.predict_total_time(validation_eval, best_params)),
        "test": base_v4.rank_metrics(test_eval, base_v4.predict_total_time(test_eval, best_params)),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(best, indent=2))
    METRICS_PATH.write_text(json.dumps(full_metrics, indent=2))

    print("\nBest front-focused thresholds:")
    print(json.dumps(best_thresholds, indent=2))
    print("\nBest front-focused cliff thresholds:")
    print(json.dumps(best_cliff_thresholds, indent=2))
    print("\nFull evaluation:")
    print(json.dumps(full_metrics, indent=2))


if __name__ == "__main__":
    main()
