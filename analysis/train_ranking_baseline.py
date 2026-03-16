#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "analysis" / "historical_driver_features.csv"
SPLIT_SUMMARY_PATH = ROOT / "analysis" / "splits" / "split_summary.json"
SPLIT_PATH = ROOT / "analysis" / "splits" / "race_id_splits.json"
OUTPUT_DIR = ROOT / "analysis" / "models" / "baseline_rank_regression"
METRICS_PATH = OUTPUT_DIR / "metrics.json"


def load_config():
    summary = json.loads(SPLIT_SUMMARY_PATH.read_text())
    split_ids = json.loads(SPLIT_PATH.read_text())
    feature_set = summary["feature_set"]
    return feature_set, split_ids


def build_model(categorical_features, numerical_features):
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
                categorical_features,
            ),
            (
                "numerical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", Ridge(alpha=10.0)),
        ]
    )


def evaluate_split(frame):
    exact = 0
    top3 = 0
    top5 = 0
    kendall_scores = []
    spearman_scores = []

    for _, race in frame.groupby("race_id", sort=False):
        predicted = race.sort_values(["pred_score", "driver_id"], ascending=[True, True])
        actual = race.sort_values(["finish_rank", "driver_id"], ascending=[True, True])

        predicted_ids = predicted["driver_id"].tolist()
        actual_ids = actual["driver_id"].tolist()
        exact += int(predicted_ids == actual_ids)
        top3 += int(set(predicted_ids[:3]) == set(actual_ids[:3]))
        top5 += int(set(predicted_ids[:5]) == set(actual_ids[:5]))

        merged = actual[["driver_id", "finish_rank"]].merge(
            predicted[["driver_id"]].assign(pred_rank=np.arange(1, len(predicted) + 1)),
            on="driver_id",
            how="inner",
        )
        kendall_scores.append(float(kendalltau(merged["finish_rank"], merged["pred_rank"]).statistic))
        spearman_scores.append(float(spearmanr(merged["finish_rank"], merged["pred_rank"]).statistic))

    race_count = frame["race_id"].nunique()
    return {
        "races": int(race_count),
        "exact_order_accuracy": exact / race_count,
        "top3_set_accuracy": top3 / race_count,
        "top5_set_accuracy": top5 / race_count,
        "mean_kendall_tau": float(np.mean(kendall_scores)),
        "mean_spearman_rho": float(np.mean(spearman_scores)),
    }


def main():
    feature_set, split_ids = load_config()
    categorical_features = feature_set["categorical_features"]
    numerical_features = feature_set["numerical_features"]
    target_column = feature_set["target_column"]

    df = pd.read_csv(DATA_PATH)
    model = build_model(categorical_features, numerical_features)

    split_frames = {}
    for split_name, race_ids in split_ids.items():
        split_frames[split_name] = df[df["race_id"].isin(race_ids)].copy()

    train = split_frames["train"]
    model.fit(train[categorical_features + numerical_features], train[target_column])

    metrics = {}
    for split_name in ("validation", "test"):
        frame = split_frames[split_name].copy()
        frame["pred_score"] = model.predict(frame[categorical_features + numerical_features])
        metrics[split_name] = evaluate_split(frame)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
