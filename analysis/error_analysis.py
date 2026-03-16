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
RACE_REPORT_PATH = OUTPUT_DIR / "validation_race_report.csv"
SEGMENT_REPORT_PATH = OUTPUT_DIR / "validation_segment_report.json"


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


def add_predictions(df, model, feature_columns):
    out = df.copy()
    out["pred_score"] = model.predict(out[feature_columns])
    return out


def build_race_report(frame):
    rows = []
    for race_id, race in frame.groupby("race_id", sort=False):
        predicted = race.sort_values(["pred_score", "driver_id"], ascending=[True, True])
        actual = race.sort_values(["finish_rank", "driver_id"], ascending=[True, True])

        predicted_ids = predicted["driver_id"].tolist()
        actual_ids = actual["driver_id"].tolist()

        merged = actual[["driver_id", "finish_rank"]].merge(
            predicted[["driver_id"]].assign(pred_rank=np.arange(1, len(predicted) + 1)),
            on="driver_id",
            how="inner",
        )
        merged["abs_rank_error"] = (merged["finish_rank"] - merged["pred_rank"]).abs()

        rows.append(
            {
                "race_id": race_id,
                "track": race["track"].iloc[0],
                "total_laps": int(race["total_laps"].iloc[0]),
                "track_temp": int(race["track_temp"].iloc[0]),
                "mean_pit_stop_count": float(race["pit_stop_count"].mean()),
                "exact_order": int(predicted_ids == actual_ids),
                "top3_set_match": int(set(predicted_ids[:3]) == set(actual_ids[:3])),
                "top5_set_match": int(set(predicted_ids[:5]) == set(actual_ids[:5])),
                "kendall_tau": float(kendalltau(merged["finish_rank"], merged["pred_rank"]).statistic),
                "spearman_rho": float(spearmanr(merged["finish_rank"], merged["pred_rank"]).statistic),
                "mean_abs_rank_error": float(merged["abs_rank_error"].mean()),
                "max_abs_rank_error": int(merged["abs_rank_error"].max()),
            }
        )

    return pd.DataFrame(rows).sort_values(["kendall_tau", "mean_abs_rank_error"], ascending=[True, False])


def summarize_segments(frame):
    summaries = {}

    frame = frame.copy()
    frame["abs_rank_error"] = (frame["finish_rank"] - frame["pred_rank"]).abs()

    for segment_col in ["track", "starting_tire", "pit_stop_count", "tire_sequence"]:
        grouped = (
            frame.groupby(segment_col)
            .agg(
                rows=("driver_id", "size"),
                mean_abs_rank_error=("abs_rank_error", "mean"),
                median_abs_rank_error=("abs_rank_error", "median"),
                mean_finish_rank=("finish_rank", "mean"),
                mean_pred_rank=("pred_rank", "mean"),
            )
            .sort_values("mean_abs_rank_error", ascending=False)
        )
        if segment_col == "tire_sequence":
            grouped = grouped.head(15)
        summaries[segment_col] = grouped.round(4).reset_index().to_dict(orient="records")

    return summaries


def main():
    feature_set, split_ids = load_config()
    categorical_features = feature_set["categorical_features"]
    numerical_features = feature_set["numerical_features"]
    feature_columns = categorical_features + numerical_features

    df = pd.read_csv(DATA_PATH)
    train = df[df["race_id"].isin(split_ids["train"])].copy()
    validation = df[df["race_id"].isin(split_ids["validation"])].copy()

    model = build_model(categorical_features, numerical_features)
    model.fit(train[feature_columns], train["finish_rank"])

    validation = add_predictions(validation, model, feature_columns)

    validation = validation.sort_values(["race_id", "pred_score", "driver_id"], ascending=[True, True, True]).copy()
    validation["pred_rank"] = validation.groupby("race_id").cumcount() + 1

    race_report = build_race_report(validation)
    segment_report = summarize_segments(validation)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    race_report.to_csv(RACE_REPORT_PATH, index=False)
    SEGMENT_REPORT_PATH.write_text(json.dumps(segment_report, indent=2))

    print("Worst races by Kendall tau:")
    print(race_report.head(10).to_string(index=False))
    print("\nWorst track segments by mean abs rank error:")
    print(pd.DataFrame(segment_report["track"]).head(10).to_string(index=False))
    print("\nWorst starting tire segments by mean abs rank error:")
    print(pd.DataFrame(segment_report["starting_tire"]).head(10).to_string(index=False))
    print("\nWorst pit stop count segments by mean abs rank error:")
    print(pd.DataFrame(segment_report["pit_stop_count"]).head(10).to_string(index=False))
    print("\nWorst tire sequence segments by mean abs rank error:")
    print(pd.DataFrame(segment_report["tire_sequence"]).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
