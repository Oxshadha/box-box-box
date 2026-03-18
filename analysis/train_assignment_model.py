#!/usr/bin/env python3
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau, spearmanr
from xgboost import XGBClassifier

from train_xgb_ranker import (
    BASE_CATEGORICAL_COLUMNS,
    BASE_FEATURE_COLUMNS,
    LEARNING_RATE,
    MAX_DEPTH,
    N_ESTIMATORS,
    TRAIN_RACES,
    USE_COMPOSITION_FEATURES,
    add_physics_predictions,
    add_race_composition_features,
    add_regime_features,
    derive_major_sequences,
    get_feature_config,
    load_inputs,
    one_hot_encode,
)


ROOT = Path(__file__).resolve().parent.parent
MODEL_TAG = os.environ.get("MODEL_TAG", "").strip()
MAX_POSITIONS = int(os.environ.get("MAX_POSITIONS", "20"))

MODEL_DIR_NAME = f"assignment_model_{MODEL_TAG}" if MODEL_TAG else "assignment_model"
OUTPUT_DIR = ROOT / "analysis" / "models" / MODEL_DIR_NAME
METRICS_PATH = OUTPUT_DIR / "metrics.json"
MODEL_PATH = OUTPUT_DIR / "model.json"
METADATA_PATH = OUTPUT_DIR / "metadata.json"


def evaluate_assignment(frame, pred_proba):
    exact = 0
    top3 = 0
    top5 = 0
    kendalls = []
    spearmans = []

    ordered_groups = list(frame.groupby("race_id", sort=False))
    offset = 0
    for _, race in ordered_groups:
        race_rows = len(race)
        probs = pred_proba[offset : offset + race_rows, :race_rows]
        offset += race_rows

        cost = -np.log(np.clip(probs, 1e-9, 1.0))
        row_ind, col_ind = linear_sum_assignment(cost)
        assigned = pd.DataFrame(
            {
                "driver_id": race.iloc[row_ind]["driver_id"].to_numpy(),
                "pred_rank": (col_ind + 1).astype(int),
            }
        )
        predicted = assigned.sort_values(["pred_rank", "driver_id"], ascending=[True, True])
        actual = race.sort_values(["finish_rank", "driver_id"], ascending=[True, True])
        pred_ids = predicted["driver_id"].tolist()
        actual_ids = actual["driver_id"].tolist()

        exact += int(pred_ids == actual_ids)
        top3 += int(set(pred_ids[:3]) == set(actual_ids[:3]))
        top5 += int(set(pred_ids[:5]) == set(actual_ids[:5]))

        merged = actual[["driver_id", "finish_rank"]].merge(assigned, on="driver_id", how="inner")
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
    df = add_regime_features(df)

    train = df[df["race_id"].isin(splits["train"])].copy()
    validation = df[df["race_id"].isin(splits["validation"])].copy()
    test = df[df["race_id"].isin(splits["test"])].copy()

    if USE_COMPOSITION_FEATURES:
        major_sequences = derive_major_sequences(train)
        df = add_race_composition_features(df, major_sequences)
        train = df[df["race_id"].isin(splits["train"])].copy()
        validation = df[df["race_id"].isin(splits["validation"])].copy()
        test = df[df["race_id"].isin(splits["test"])].copy()
        feature_columns, categorical_columns = get_feature_config(major_sequences)
    else:
        major_sequences = []
        feature_columns = list(BASE_FEATURE_COLUMNS)
        categorical_columns = list(BASE_CATEGORICAL_COLUMNS)

    available_races = train["race_id"].drop_duplicates()
    sampled_train_races = available_races.sample(
        n=min(TRAIN_RACES, len(available_races)),
        random_state=42,
    )
    train = train[train["race_id"].isin(sampled_train_races)].copy()

    train_x, validation_x, test_x, categories, encoded_columns = one_hot_encode(
        train[feature_columns],
        validation[feature_columns],
        test[feature_columns],
        categorical_columns,
    )

    train_y = (train["finish_rank"].astype(int) - 1).clip(lower=0, upper=MAX_POSITIONS - 1)
    validation_y = (validation["finish_rank"].astype(int) - 1).clip(lower=0, upper=MAX_POSITIONS - 1)
    test_y = (test["finish_rank"].astype(int) - 1).clip(lower=0, upper=MAX_POSITIONS - 1)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=MAX_POSITIONS,
        tree_method="hist",
        learning_rate=LEARNING_RATE,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=1,
        random_state=42,
    )

    model.fit(
        train_x,
        train_y,
        eval_set=[(validation_x, validation_y)],
        verbose=False,
    )

    validation_proba = model.predict_proba(validation_x)
    test_proba = model.predict_proba(test_x)

    metrics = {
        "validation": {
            "assignment_model": evaluate_assignment(validation, validation_proba),
        },
        "test": {
            "assignment_model": evaluate_assignment(test, test_proba),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    model.save_model(str(MODEL_PATH))
    METADATA_PATH.write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "categorical_columns": categorical_columns,
                "categories": categories,
                "encoded_columns": encoded_columns,
                "major_sequences": major_sequences,
                "use_composition_features": USE_COMPOSITION_FEATURES,
                "model_tag": MODEL_TAG,
                "train_races": len(sampled_train_races),
                "n_estimators": N_ESTIMATORS,
                "learning_rate": LEARNING_RATE,
                "max_depth": MAX_DEPTH,
                "max_positions": MAX_POSITIONS,
            },
            indent=2,
        )
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
