#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "analysis" / "historical_driver_features.csv"
OUTPUT_DIR = ROOT / "analysis" / "splits"
SPLIT_PATH = OUTPUT_DIR / "race_id_splits.json"
SUMMARY_PATH = OUTPUT_DIR / "split_summary.json"
RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "track",
    "starting_tire",
    "tire_sequence",
    "stint_1_compound",
    "stint_2_compound",
    "stint_3_compound",
]

NUMERICAL_FEATURES = [
    "total_laps",
    "base_lap_time",
    "pit_lane_time",
    "track_temp",
    "pit_stop_count",
    "first_stop_lap",
    "second_stop_lap",
    "stint_count",
    "soft_laps",
    "soft_stints",
    "soft_max_stint",
    "medium_laps",
    "medium_stints",
    "medium_max_stint",
    "hard_laps",
    "hard_stints",
    "hard_max_stint",
    "stint_1_laps",
    "stint_2_laps",
    "stint_3_laps",
]

EXCLUDED_COLUMNS = [
    "race_id",
    "grid_slot",
    "driver_id",
    "finish_rank",
    "finish_rank_pct",
]

TARGET_COLUMN = "finish_rank"


def group_split(df: pd.DataFrame):
    race_frame = df[["race_id"]].drop_duplicates().reset_index(drop=True)
    groups = race_frame["race_id"]

    outer = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=RANDOM_STATE)
    train_idx, holdout_idx = next(outer.split(race_frame, groups=groups))
    train_races = race_frame.iloc[train_idx]["race_id"].tolist()
    holdout_races = race_frame.iloc[holdout_idx]["race_id"].tolist()

    holdout_frame = race_frame.iloc[holdout_idx].reset_index(drop=True)
    inner = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=RANDOM_STATE)
    val_idx, test_idx = next(inner.split(holdout_frame, groups=holdout_frame["race_id"]))
    val_races = holdout_frame.iloc[val_idx]["race_id"].tolist()
    test_races = holdout_frame.iloc[test_idx]["race_id"].tolist()

    return {
        "train": train_races,
        "validation": val_races,
        "test": test_races,
    }


def split_stats(df: pd.DataFrame, race_ids):
    part = df[df["race_id"].isin(race_ids)].copy()
    return {
        "races": int(part["race_id"].nunique()),
        "rows": int(len(part)),
        "tracks": part["track"].value_counts().sort_index().to_dict(),
        "starting_tire_pct": (
            part["starting_tire"].value_counts(normalize=True).sort_index().mul(100).round(2).to_dict()
        ),
        "pit_stop_count_pct": (
            part["pit_stop_count"].value_counts(normalize=True).sort_index().mul(100).round(2).to_dict()
        ),
        "track_temp_range": [int(part["track_temp"].min()), int(part["track_temp"].max())],
        "total_laps_range": [int(part["total_laps"].min()), int(part["total_laps"].max())],
    }


def main():
    df = pd.read_csv(DATA_PATH)
    splits = group_split(df)

    feature_set = {
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "excluded_columns": EXCLUDED_COLUMNS,
        "target_column": TARGET_COLUMN,
    }

    summary = {
        "feature_set": feature_set,
        "splits": {
            split_name: split_stats(df, race_ids)
            for split_name, race_ids in splits.items()
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(splits, indent=2))
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
