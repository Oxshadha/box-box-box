#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_physics_simulator_v2 as sim_v2
import calibrate_physics_simulator_v4 as sim_v4


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "analysis" / "historical_driver_features.csv"
SPLIT_PATH = ROOT / "analysis" / "splits" / "race_id_splits.json"
V2_PARAMS_PATH = ROOT / "analysis" / "models" / "physics_simulator_v2" / "best_params.json"
V4_PARAMS_PATH = ROOT / "analysis" / "models" / "physics_simulator_v4" / "best_params.json"
OUTPUT_DIR = ROOT / "analysis" / "models" / "physics_blend_v2_v4"
RESULTS_PATH = OUTPUT_DIR / "blend_search.json"


def load_inputs():
    df = pd.read_csv(DATA_PATH)
    splits = json.loads(SPLIT_PATH.read_text())
    v2 = json.loads(V2_PARAMS_PATH.read_text())
    v4 = json.loads(V4_PARAMS_PATH.read_text())
    return df, splits, v2, v4


def prepare_predictions(df, splits, v2_record, v4_record):
    data = {}
    for split_name in ("validation", "test"):
        part = df[df["race_id"].isin(splits[split_name])].copy()

        part_v2 = sim_v2.add_piecewise_statistics(part, v2_record["thresholds"])
        part["pred_v2"] = sim_v2.predict_total_time(part_v2, v2_record["params"])

        part_v4 = sim_v4.add_piecewise_statistics(
            part,
            v4_record["thresholds"],
            v4_record["cliff_thresholds"],
        )
        part["pred_v4"] = sim_v4.predict_total_time(part_v4, v4_record["params"])
        data[split_name] = part
    return data


def blend_metrics(frame, alpha):
    scored = frame[["race_id", "driver_id", "finish_rank"]].copy()
    scored["pred_score"] = alpha * frame["pred_v2"] + (1.0 - alpha) * frame["pred_v4"]
    return sim_v2.rank_metrics(scored, scored["pred_score"].to_numpy())


def selection_score(metrics):
    return 0.6 * metrics["top3_set_accuracy"] + 0.4 * metrics["top5_set_accuracy"]


def main():
    df, splits, v2_record, v4_record = load_inputs()
    split_frames = prepare_predictions(df, splits, v2_record, v4_record)
    sampled_validation_races = (
        split_frames["validation"]["race_id"].drop_duplicates().sample(n=500, random_state=42)
    )
    validation_search_frame = split_frames["validation"][
        split_frames["validation"]["race_id"].isin(sampled_validation_races)
    ].copy()

    candidates = []
    for alpha in np.linspace(0.0, 1.0, 6):
        validation_metrics = blend_metrics(validation_search_frame, float(alpha))
        candidates.append(
            {
                "alpha_v2": float(alpha),
                "alpha_v4": float(1.0 - alpha),
                "validation_metrics": validation_metrics,
                "selection_score": selection_score(validation_metrics),
            }
        )

    best = sorted(
        candidates,
        key=lambda item: (
            item["selection_score"],
            item["validation_metrics"]["top3_set_accuracy"],
            item["validation_metrics"]["top5_set_accuracy"],
            item["validation_metrics"]["mean_kendall_tau"],
        ),
        reverse=True,
    )[0]

    test_metrics = blend_metrics(split_frames["test"], best["alpha_v2"])
    result = {
        "best_blend": best,
        "test_metrics": test_metrics,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2))

    print("Top blend candidates by validation score:")
    for item in sorted(candidates, key=lambda x: x["selection_score"], reverse=True)[:5]:
        vm = item["validation_metrics"]
        print(
            json.dumps(
                {
                    "alpha_v2": item["alpha_v2"],
                    "alpha_v4": item["alpha_v4"],
                    "selection_score": round(item["selection_score"], 6),
                    "top3": round(vm["top3_set_accuracy"], 6),
                    "top5": round(vm["top5_set_accuracy"], 6),
                    "kendall": round(vm["mean_kendall_tau"], 6),
                    "spearman": round(vm["mean_spearman_rho"], 6),
                }
            )
        )

    print("\nBest blend on validation:")
    print(json.dumps(best, indent=2))
    print("\nHeld-out test metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
