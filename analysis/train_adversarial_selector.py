#!/usr/bin/env python3
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import train_xgb_ranker as base


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "test_cases" / "inputs"
OUTPUT_DIR = ROOT / "analysis" / "adversarial_selection"
TOP_RACES = int(os.environ.get("TOP_RACES", "2500"))

RACE_NUMERIC = [
    "total_laps",
    "track_temp",
    "pit_lane_time",
    "base_lap_time",
    "unique_sequence_count",
    "two_stop_drivers",
    "top_sequence_count",
    "soft_share",
    "medium_share",
    "hard_share",
    "soft_hard_share",
    "medium_hard_share",
    "hard_medium_share",
    "medium_soft_share",
    "soft_medium_share",
]
RACE_CATEGORICAL = ["track", "lap_band", "temp_band", "top_sequence"]


def summarize_historical_races(df):
    rows = []
    for race_id, race in df.groupby("race_id", sort=False):
        seq_counts = race["tire_sequence"].value_counts()
        total = float(len(race))
        rows.append(
            {
                "race_id": race_id,
                "track": str(race["track"].iloc[0]),
                "total_laps": float(race["total_laps"].iloc[0]),
                "track_temp": float(race["track_temp"].iloc[0]),
                "pit_lane_time": float(race["pit_lane_time"].iloc[0]),
                "base_lap_time": float(race["base_lap_time"].iloc[0]),
                "lap_band": str(race["lap_band"].iloc[0]),
                "temp_band": str(race["temp_band"].iloc[0]),
                "unique_sequence_count": float(race["tire_sequence"].nunique()),
                "two_stop_drivers": float((race["pit_stop_count"] == 2).sum()),
                "top_sequence": str(seq_counts.index[0]),
                "top_sequence_count": float(seq_counts.iloc[0]),
                "soft_share": float((race["starting_tire"] == "SOFT").sum()) / total,
                "medium_share": float((race["starting_tire"] == "MEDIUM").sum()) / total,
                "hard_share": float((race["starting_tire"] == "HARD").sum()) / total,
                "soft_hard_share": float((race["tire_sequence"] == "SOFT>HARD").sum()) / total,
                "medium_hard_share": float((race["tire_sequence"] == "MEDIUM>HARD").sum()) / total,
                "hard_medium_share": float((race["tire_sequence"] == "HARD>MEDIUM").sum()) / total,
                "medium_soft_share": float((race["tire_sequence"] == "MEDIUM>SOFT").sum()) / total,
                "soft_medium_share": float((race["tire_sequence"] == "SOFT>MEDIUM").sum()) / total,
            }
        )
    return pd.DataFrame(rows)


def summarize_challenge_inputs():
    rows = []
    for input_path in sorted(INPUT_DIR.glob("test_*.json")):
        race = json.loads(input_path.read_text())
        strategies = list(race["strategies"].values())
        seqs = []
        starts = []
        two_stop = 0
        for strategy in strategies:
            starts.append(strategy["starting_tire"])
            seqs.append(">".join([strategy["starting_tire"]] + [stop["to_tire"] for stop in strategy["pit_stops"]]))
            if len(strategy["pit_stops"]) == 2:
                two_stop += 1
        seq_counts = pd.Series(seqs).value_counts()
        total = float(len(strategies))
        track_temp = float(race["race_config"]["track_temp"])
        total_laps = float(race["race_config"]["total_laps"])
        if track_temp <= 24:
            temp_band = "cool"
        elif track_temp <= 30:
            temp_band = "mild"
        elif track_temp <= 36:
            temp_band = "warm"
        else:
            temp_band = "hot"
        if total_laps <= 35:
            lap_band = "short"
        elif total_laps <= 45:
            lap_band = "mid_short"
        elif total_laps <= 55:
            lap_band = "mid"
        else:
            lap_band = "long"
        rows.append(
            {
                "race_id": race["race_id"],
                "track": str(race["race_config"]["track"]),
                "total_laps": total_laps,
                "track_temp": track_temp,
                "pit_lane_time": float(race["race_config"]["pit_lane_time"]),
                "base_lap_time": float(race["race_config"]["base_lap_time"]),
                "lap_band": lap_band,
                "temp_band": temp_band,
                "unique_sequence_count": float(len(set(seqs))),
                "two_stop_drivers": float(two_stop),
                "top_sequence": str(seq_counts.index[0]),
                "top_sequence_count": float(seq_counts.iloc[0]),
                "soft_share": starts.count("SOFT") / total,
                "medium_share": starts.count("MEDIUM") / total,
                "hard_share": starts.count("HARD") / total,
                "soft_hard_share": seqs.count("SOFT>HARD") / total,
                "medium_hard_share": seqs.count("MEDIUM>HARD") / total,
                "hard_medium_share": seqs.count("HARD>MEDIUM") / total,
                "medium_soft_share": seqs.count("MEDIUM>SOFT") / total,
                "soft_medium_share": seqs.count("SOFT>MEDIUM") / total,
            }
        )
    return pd.DataFrame(rows)


def build_model():
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "categorical",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            RACE_CATEGORICAL,
                        ),
                        (
                            "numerical",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            RACE_NUMERIC,
                        ),
                    ]
                ),
            ),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


def main():
    df, splits, v2_record, v4_record = base.load_inputs()
    df = base.add_physics_predictions(df, v2_record, v4_record)
    df = base.add_regime_features(df)

    historical = summarize_historical_races(df)
    challenge = summarize_challenge_inputs()

    historical["is_challenge"] = 0
    challenge["is_challenge"] = 1

    train_hist = historical[historical["race_id"].isin(splits["train"])].copy()
    holdout_hist = historical[~historical["race_id"].isin(splits["train"])].copy()
    combined_train = pd.concat([train_hist, challenge], ignore_index=True)

    model = build_model()
    model.fit(combined_train[RACE_CATEGORICAL + RACE_NUMERIC], combined_train["is_challenge"])

    train_hist["challenge_score"] = model.predict_proba(train_hist[RACE_CATEGORICAL + RACE_NUMERIC])[:, 1]
    holdout_eval = pd.concat([holdout_hist, challenge], ignore_index=True)
    holdout_proba = model.predict_proba(holdout_eval[RACE_CATEGORICAL + RACE_NUMERIC])[:, 1]
    holdout_auc = roc_auc_score(holdout_eval["is_challenge"], holdout_proba)

    ranked = train_hist.sort_values(["challenge_score", "race_id"], ascending=[False, True]).reset_index(drop=True)
    top_races = ranked["race_id"].head(min(TOP_RACES, len(ranked))).tolist()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "top_train_races.json").write_text(json.dumps({"train_races": top_races}, indent=2))
    (OUTPUT_DIR / "train_race_scores.csv").write_text(ranked.to_csv(index=False))
    metrics = {
        "challenge_cases": int(len(challenge)),
        "train_historical_races": int(len(train_hist)),
        "holdout_auc": float(holdout_auc),
        "top_races": int(min(TOP_RACES, len(ranked))),
        "top_score_mean": float(ranked["challenge_score"].head(min(TOP_RACES, len(ranked))).mean()),
        "all_score_mean": float(ranked["challenge_score"].mean()),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
