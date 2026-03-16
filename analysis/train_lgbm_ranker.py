#!/usr/bin/env python3
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

import calibrate_physics_simulator_v2 as sim_v2
import calibrate_physics_simulator_v4 as sim_v4


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "analysis" / "historical_driver_features.csv"
SPLIT_PATH = ROOT / "analysis" / "splits" / "race_id_splits.json"
V2_PARAMS_PATH = ROOT / "analysis" / "models" / "physics_simulator_v2" / "best_params.json"
V4_PARAMS_PATH = ROOT / "analysis" / "models" / "physics_simulator_v4" / "best_params.json"
OUTPUT_DIR = ROOT / "analysis" / "models" / "lgbm_ranker"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
MODEL_PATH = OUTPUT_DIR / "model.txt"
METADATA_PATH = OUTPUT_DIR / "metadata.json"

TRAIN_RACES = int(os.environ.get("TRAIN_RACES", "600"))
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", "300"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.05"))
NUM_LEAVES = int(os.environ.get("NUM_LEAVES", "31"))
MIN_CHILD_SAMPLES = int(os.environ.get("MIN_CHILD_SAMPLES", "50"))


CATEGORICAL_COLUMNS = [
    "track",
    "starting_tire",
    "tire_sequence",
    "stint_1_compound",
    "stint_2_compound",
    "stint_3_compound",
]

NUMERIC_COLUMNS = [
    "pred_v2",
    "pred_v4",
    "pred_blend",
    "pred_gap_v2_v4",
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
    out["pred_gap_v2_v4"] = pred_v2 - pred_v4
    return out


def encode_frame(frame):
    out = frame.copy()
    categories = {}
    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].fillna("__MISSING__").astype("category")
        categories[col] = out[col].cat.categories.tolist()
    for col in NUMERIC_COLUMNS:
        out[col] = out[col].fillna(out[col].median())
    return out, categories


def apply_categories(frame, categories):
    out = frame.copy()
    for col in CATEGORICAL_COLUMNS:
        out[col] = pd.Categorical(
            out[col].fillna("__MISSING__"),
            categories=categories[col],
        )
    for col in NUMERIC_COLUMNS:
        out[col] = out[col].fillna(out[col].median())
    return out


def group_sizes(frame):
    return frame.groupby("race_id", sort=False).size().tolist()


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

    train = df[df["race_id"].isin(splits["train"])].copy()
    validation = df[df["race_id"].isin(splits["validation"])].copy()
    test = df[df["race_id"].isin(splits["test"])].copy()

    # Keep first pass tractable but still broad.
    available_races = train["race_id"].drop_duplicates()
    sampled_train_races = available_races.sample(
        n=min(TRAIN_RACES, len(available_races)),
        random_state=42,
    )
    train = train[train["race_id"].isin(sampled_train_races)].copy()

    train, categories = encode_frame(train)
    validation = apply_categories(validation, categories)
    test = apply_categories(test, categories)

    feature_columns = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
    categorical_feature = CATEGORICAL_COLUMNS

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        learning_rate=LEARNING_RATE,
        n_estimators=N_ESTIMATORS,
        num_leaves=NUM_LEAVES,
        min_child_samples=MIN_CHILD_SAMPLES,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=1,
        force_row_wise=True,
        random_state=42,
    )

    model.fit(
        train[feature_columns],
        train["finish_rank"],
        group=group_sizes(train),
        eval_set=[(validation[feature_columns], validation["finish_rank"])],
        eval_group=[group_sizes(validation)],
        eval_at=[3, 5, 10],
        categorical_feature=categorical_feature,
    )

    metrics = {
        "validation": {
            "physics_blend": evaluate(validation, validation["pred_blend"].to_numpy()),
            "lgbm_ranker": evaluate(validation, model.predict(validation[feature_columns])),
        },
        "test": {
            "physics_blend": evaluate(test, test["pred_blend"].to_numpy()),
            "lgbm_ranker": evaluate(test, model.predict(test[feature_columns])),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    model.booster_.save_model(str(MODEL_PATH))
    METADATA_PATH.write_text(
        json.dumps(
            {
                "train_races": len(sampled_train_races),
                "n_estimators": N_ESTIMATORS,
                "learning_rate": LEARNING_RATE,
                "num_leaves": NUM_LEAVES,
                "min_child_samples": MIN_CHILD_SAMPLES,
                "categorical_columns": CATEGORICAL_COLUMNS,
                "numeric_columns": NUMERIC_COLUMNS,
                "categories": categories,
            },
            indent=2,
        )
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
