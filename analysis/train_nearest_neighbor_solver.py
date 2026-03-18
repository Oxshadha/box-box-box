#!/usr/bin/env python3
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from nearest_neighbor_solver_lib import load_dataset, nearest_order, save_artifact, build_artifact


ROOT = Path(__file__).resolve().parent.parent
TRAIN_RACES = int(os.environ.get("TRAIN_RACES", "6000"))
TOP_K = int(os.environ.get("TOP_K", "7"))
MODEL_TAG = os.environ.get("MODEL_TAG", "").strip()
MODEL_DIR_NAME = f"nearest_neighbor_{MODEL_TAG}" if MODEL_TAG else "nearest_neighbor"
OUTPUT_DIR = ROOT / "analysis" / "models" / MODEL_DIR_NAME


def evaluate(df, race_ids, artifact):
    exact = 0
    top3 = 0
    top5 = 0
    races = 0
    for race_id in race_ids:
        race = df[df["race_id"] == race_id].copy()
        config = {
            "race_id": race_id,
            "race_config": {
                "track": race["track"].iloc[0],
                "total_laps": int(race["total_laps"].iloc[0]),
                "track_temp": float(race["track_temp"].iloc[0]),
                "pit_lane_time": float(race["pit_lane_time"].iloc[0]),
                "base_lap_time": float(race["base_lap_time"].iloc[0]),
            },
            "strategies": {},
        }
        for _, row in race.iterrows():
            stops = []
            if int(row["pit_stop_count"]) >= 1 and not np.isnan(row["first_stop_lap"]):
                next_tire = row["stint_2_compound"] if pd.notna(row["stint_2_compound"]) else row["starting_tire"]
                stops.append({"lap": int(row["first_stop_lap"]), "to_tire": str(next_tire)})
            if int(row["pit_stop_count"]) >= 2 and not np.isnan(row["second_stop_lap"]):
                next_tire = row["stint_3_compound"] if pd.notna(row["stint_3_compound"]) else row["starting_tire"]
                stops.append({"lap": int(row["second_stop_lap"]), "to_tire": str(next_tire)})
            config["strategies"][row["grid_slot"]] = {
                "driver_id": row["driver_id"],
                "starting_tire": row["starting_tire"],
                "pit_stops": stops,
            }
        pred = nearest_order(config, artifact)
        actual = race.sort_values(["finish_rank", "driver_id"], ascending=[True, True])["driver_id"].tolist()
        exact += int(pred == actual)
        top3 += int(set(pred[:3]) == set(actual[:3]))
        top5 += int(set(pred[:5]) == set(actual[:5]))
        races += 1
    return {
        "races": races,
        "exact_order_accuracy": exact / races if races else 0.0,
        "top3_set_accuracy": top3 / races if races else 0.0,
        "top5_set_accuracy": top5 / races if races else 0.0,
    }


def main():
    df, splits = load_dataset()
    train_ids = df[df["race_id"].isin(splits["train"])]["race_id"].drop_duplicates()
    sampled_train_ids = train_ids.sample(n=min(TRAIN_RACES, len(train_ids)), random_state=42).tolist()
    artifact = build_artifact(df, sampled_train_ids, top_k=TOP_K)
    metrics = {
        "validation": evaluate(df, splits["validation"], artifact),
        "test": evaluate(df, splits["test"], artifact),
        "config": {
            "train_races": len(sampled_train_ids),
            "top_k": TOP_K,
        },
    }
    save_artifact(OUTPUT_DIR, artifact, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
