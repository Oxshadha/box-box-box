#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

import calibrate_physics_simulator_v2 as sim_v2
import calibrate_physics_simulator_v4 as sim_v4


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "analysis" / "historical_driver_features.csv"
SPLIT_PATH = ROOT / "analysis" / "splits" / "race_id_splits.json"
V2_PARAMS_PATH = ROOT / "analysis" / "models" / "physics_simulator_v2" / "best_params.json"
V4_PARAMS_PATH = ROOT / "analysis" / "models" / "physics_simulator_v4" / "best_params.json"
OUTPUT_DIR = ROOT / "analysis" / "models" / "hybrid_histgb"
METRICS_PATH = OUTPUT_DIR / "metrics.json"


CATEGORICAL_FEATURES = [
    "track",
    "starting_tire",
    "tire_sequence",
    "stint_1_compound",
    "stint_2_compound",
    "stint_3_compound",
]

NUMERICAL_FEATURES = [
    "pred_v2",
    "pred_v4",
    "pred_blend",
    "total_laps",
    "base_lap_time",
    "pit_lane_time",
    "track_temp",
    "pit_stop_count",
    "first_stop_lap",
    "second_stop_lap",
    "soft_laps",
    "medium_laps",
    "hard_laps",
    "soft_max_stint",
    "medium_max_stint",
    "hard_max_stint",
    "stint_1_laps",
    "stint_2_laps",
    "stint_3_laps",
]


def load_inputs():
    df = pd.read_csv(DATA_PATH)
    splits = json.loads(SPLIT_PATH.read_text())
    v2 = json.loads(V2_PARAMS_PATH.read_text())
    v4 = json.loads(V4_PARAMS_PATH.read_text())
    return df, splits, v2, v4


def add_physics_predictions(df, v2_record, v4_record):
    part_v2 = sim_v2.add_piecewise_statistics(df.copy(), v2_record["thresholds"])
    pred_v2 = sim_v2.predict_total_time(part_v2, v2_record["params"])

    part_v4 = sim_v4.add_piecewise_statistics(
        df.copy(),
        v4_record["thresholds"],
        v4_record["cliff_thresholds"],
    )
    pred_v4 = sim_v4.predict_total_time(part_v4, v4_record["params"])

    out = df.copy()
    out["pred_v2"] = pred_v2
    out["pred_v4"] = pred_v4
    out["pred_blend"] = 0.6 * pred_v2 + 0.4 * pred_v4
    return out


def encode_categories(df):
    out = df.copy()
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].fillna("__MISSING__").astype("category")
        out[col] = out[col].cat.codes
    return out


def evaluate(frame, pred_score):
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

    race_count = frame["race_id"].nunique()
    return {
        "races": int(race_count),
        "exact_order_accuracy": exact / race_count,
        "top3_set_accuracy": top3 / race_count,
        "top5_set_accuracy": top5 / race_count,
        "mean_kendall_tau": float(np.mean(kendalls)),
        "mean_spearman_rho": float(np.mean(spearmans)),
    }


def main():
    df, splits, v2_record, v4_record = load_inputs()
    df = add_physics_predictions(df, v2_record, v4_record)
    df = encode_categories(df)

    feature_columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    categorical_mask = [col in CATEGORICAL_FEATURES for col in feature_columns]

    train = df[df["race_id"].isin(splits["train"])].copy()
    validation = df[df["race_id"].isin(splits["validation"])].copy()
    test = df[df["race_id"].isin(splits["test"])].copy()

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,
        categorical_features=categorical_mask,
        random_state=42,
    )

    model.fit(train[feature_columns], train["finish_rank"])

    metrics = {}
    for split_name, frame in (("validation", validation), ("test", test)):
        metrics[split_name] = {
            "physics_blend": evaluate(frame, frame["pred_blend"].to_numpy()),
            "hybrid_histgb": evaluate(frame, model.predict(frame[feature_columns])),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
