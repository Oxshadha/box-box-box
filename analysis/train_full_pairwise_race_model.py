#!/usr/bin/env python3
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import full_pairwise_model_lib as lib
import train_xgb_ranker as base


ROOT = Path(__file__).resolve().parent.parent
TRAIN_RACES = int(os.environ.get("TRAIN_RACES", "6000"))
MAX_PAIRS = int(os.environ.get("MAX_PAIRS", "300000"))


def build_pair_dataset(frame):
    rows = []
    for _, race in frame.groupby("race_id", sort=False):
        race = lib.enrich_race_with_gate(race)
        n = len(race)
        for i in range(n - 1):
            for j in range(i + 1, n):
                left = race.iloc[i]
                right = race.iloc[j]
                item = lib.pair_feature_row(left, right)
                item["y"] = int(left["finish_rank"] < right["finish_rank"])
                rows.append(item)
                reverse = lib.pair_feature_row(right, left)
                reverse["y"] = 1 - item["y"]
                rows.append(reverse)
    pair_df = pd.DataFrame(rows)
    if len(pair_df) > MAX_PAIRS:
        pair_df = pair_df.sample(n=MAX_PAIRS, random_state=42).reset_index(drop=True)
    return pair_df


def build_model():
    preprocess = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                lib.PAIR_CATEGORICAL,
            ),
            (
                "numerical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                lib.PAIR_NUMERICAL,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=2500, tol=1e-3, random_state=42)),
        ]
    )


def evaluate(frame, model):
    exact = 0
    top3 = 0
    top5 = 0
    kendalls = []
    spearmans = []
    race_count = 0
    for _, race in frame.groupby("race_id", sort=False):
        ranked = lib.rank_race_with_pairwise_model(race, model)
        pred_ids = ranked["driver_id"].tolist()
        actual = race.sort_values(["finish_rank", "driver_id"], ascending=[True, True])
        actual_ids = actual["driver_id"].tolist()
        exact += int(pred_ids == actual_ids)
        top3 += int(set(pred_ids[:3]) == set(actual_ids[:3]))
        top5 += int(set(pred_ids[:5]) == set(actual_ids[:5]))
        merged = actual[["driver_id", "finish_rank"]].merge(
            ranked[["driver_id"]].assign(pred_rank=np.arange(1, len(ranked) + 1)),
            on="driver_id",
            how="inner",
        )
        kendalls.append(float(kendalltau(merged["finish_rank"], merged["pred_rank"]).statistic))
        spearmans.append(float(spearmanr(merged["finish_rank"], merged["pred_rank"]).statistic))
        race_count += 1
    return {
        "races": race_count,
        "exact_order_accuracy": exact / race_count if race_count else 0.0,
        "top3_set_accuracy": top3 / race_count if race_count else 0.0,
        "top5_set_accuracy": top5 / race_count if race_count else 0.0,
        "mean_kendall_tau": float(np.mean(kendalls)) if kendalls else 0.0,
        "mean_spearman_rho": float(np.mean(spearmans)) if spearmans else 0.0,
    }


def main():
    df, splits, v2_record, v4_record = base.load_inputs()
    df = base.add_physics_predictions(df, v2_record, v4_record)
    df = base.add_regime_features(df)
    major_sequences = base.derive_major_sequences(df[df["race_id"].isin(splits["train"])].copy())
    df = base.add_race_composition_features(df, major_sequences)

    train = df[df["race_id"].isin(splits["train"])].copy()
    validation = df[df["race_id"].isin(splits["validation"])].copy()
    test = df[df["race_id"].isin(splits["test"])].copy()

    available_races = train["race_id"].drop_duplicates()
    sampled_train_races = base.select_train_races(available_races)
    train = train[train["race_id"].isin(sampled_train_races)].copy()

    train_pairs = build_pair_dataset(train)
    validation_pairs = build_pair_dataset(validation)

    model = build_model()
    model.fit(train_pairs[lib.PAIR_CATEGORICAL + lib.PAIR_NUMERICAL], train_pairs["y"])

    metrics = {
        "validation": {
            "pairwise_model": evaluate(validation, model),
        },
        "test": {
            "pairwise_model": evaluate(test, model),
        },
        "pair_rows": {
            "train": int(len(train_pairs)),
            "validation": int(len(validation_pairs)),
        },
        "config": {
            "train_races": int(len(sampled_train_races)),
            "max_pairs": MAX_PAIRS,
        },
    }

    lib.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with lib.MODEL_PATH.open("wb") as fh:
        import pickle

        pickle.dump(model, fh)
    lib.METADATA_PATH.write_text(
        json.dumps(
            {
                "categorical_columns": lib.PAIR_CATEGORICAL,
                "numerical_columns": lib.PAIR_NUMERICAL,
                "train_races": int(len(sampled_train_races)),
                "max_pairs": MAX_PAIRS,
            },
            indent=2,
        )
    )
    lib.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
