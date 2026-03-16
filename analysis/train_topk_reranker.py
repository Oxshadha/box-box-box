#!/usr/bin/env python3
import itertools
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import calibrate_physics_simulator_v2 as sim_v2
import calibrate_physics_simulator_v4 as sim_v4


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "analysis" / "historical_driver_features.csv"
SPLIT_PATH = ROOT / "analysis" / "splits" / "race_id_splits.json"
V2_PARAMS_PATH = ROOT / "analysis" / "models" / "physics_simulator_v2" / "best_params.json"
V4_PARAMS_PATH = ROOT / "analysis" / "models" / "physics_simulator_v4" / "best_params.json"
LGBM_MODEL_DIR = ROOT / "analysis" / "models" / "lgbm_ranker"
LGBM_MODEL_PATH = LGBM_MODEL_DIR / "model.txt"
LGBM_METADATA_PATH = LGBM_MODEL_DIR / "metadata.json"
OUTPUT_DIR = ROOT / "analysis" / "models" / "topk_reranker"
METRICS_PATH = OUTPUT_DIR / "metrics.json"


TOP_K = int(os.environ.get("TOP_K", "8"))
FRONT_GAP = float(os.environ.get("FRONT_GAP", "0.8"))
TRAIN_RACES = int(os.environ.get("TRAIN_RACES", "4000"))

PAIR_CATEGORICAL = [
    "track",
    "left_starting_tire",
    "right_starting_tire",
    "left_tire_sequence",
    "right_tire_sequence",
]

PAIR_NUMERICAL = [
    "left_rank",
    "right_rank",
    "lgbm_gap",
    "abs_lgbm_gap",
    "blend_gap",
    "abs_blend_gap",
    "left_pit_stop_count",
    "right_pit_stop_count",
    "pit_stop_gap",
    "left_first_stop_lap",
    "right_first_stop_lap",
    "first_stop_gap",
    "track_temp",
    "total_laps",
    "pit_lane_time",
]


def load_inputs():
    df = pd.read_csv(DATA_PATH)
    splits = json.loads(SPLIT_PATH.read_text())
    v2 = json.loads(V2_PARAMS_PATH.read_text())
    v4 = json.loads(V4_PARAMS_PATH.read_text())
    metadata = json.loads(LGBM_METADATA_PATH.read_text())
    booster = lgb.Booster(model_file=str(LGBM_MODEL_PATH))
    return df, splits, v2, v4, metadata, booster


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


def apply_lgbm_categories(frame, metadata):
    out = frame.copy()
    for col in metadata["categorical_columns"]:
        out[col] = pd.Categorical(
            out[col].fillna("__MISSING__"),
            categories=metadata["categories"][col],
        )
    for col in metadata["numeric_columns"]:
        if out[col].isna().any():
            non_null = out[col].dropna()
            fill_value = float(non_null.median()) if not non_null.empty else 0.0
            out[col] = out[col].fillna(fill_value)
    return out


def add_lgbm_scores(df, metadata, booster):
    encoded = apply_lgbm_categories(df.copy(), metadata)
    feature_columns = metadata["categorical_columns"] + metadata["numeric_columns"]
    out = df.copy()
    out["pred_lgbm"] = booster.predict(encoded[feature_columns])
    return out


def build_pair_dataset(frame):
    rows = []
    for _, race in frame.groupby("race_id", sort=False):
        ranked = race.sort_values(["pred_lgbm", "driver_id"], ascending=[True, True]).reset_index(drop=True)
        head = ranked.head(TOP_K).copy()
        true_rank = {row["driver_id"]: row["finish_rank"] for _, row in head.iterrows()}

        for i, j in itertools.combinations(range(len(head)), 2):
            left = head.iloc[i]
            right = head.iloc[j]
            lgbm_gap = float(left["pred_lgbm"] - right["pred_lgbm"])
            if abs(lgbm_gap) > FRONT_GAP:
                continue

            rows.append(
                {
                    "track": left["track"],
                    "left_starting_tire": left["starting_tire"],
                    "right_starting_tire": right["starting_tire"],
                    "left_tire_sequence": left["tire_sequence"],
                    "right_tire_sequence": right["tire_sequence"],
                    "left_rank": i + 1,
                    "right_rank": j + 1,
                    "lgbm_gap": lgbm_gap,
                    "abs_lgbm_gap": abs(lgbm_gap),
                    "blend_gap": float(left["pred_blend"] - right["pred_blend"]),
                    "abs_blend_gap": abs(float(left["pred_blend"] - right["pred_blend"])),
                    "left_pit_stop_count": left["pit_stop_count"],
                    "right_pit_stop_count": right["pit_stop_count"],
                    "pit_stop_gap": left["pit_stop_count"] - right["pit_stop_count"],
                    "left_first_stop_lap": left["first_stop_lap"],
                    "right_first_stop_lap": right["first_stop_lap"],
                    "first_stop_gap": left["first_stop_lap"] - right["first_stop_lap"],
                    "track_temp": left["track_temp"],
                    "total_laps": left["total_laps"],
                    "pit_lane_time": left["pit_lane_time"],
                    "y": int(true_rank[left["driver_id"]] < true_rank[right["driver_id"]]),
                    "baseline_left_ahead": int(lgbm_gap < 0),
                }
            )

    return pd.DataFrame(rows)


def build_pair_model():
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
                PAIR_CATEGORICAL,
            ),
            (
                "numerical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                PAIR_NUMERICAL,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=1e-4,
                    max_iter=2000,
                    tol=1e-3,
                    random_state=42,
                ),
            ),
        ]
    )


def pair_features(left, right, left_rank, right_rank):
    return {
        "track": left["track"],
        "left_starting_tire": left["starting_tire"],
        "right_starting_tire": right["starting_tire"],
        "left_tire_sequence": left["tire_sequence"],
        "right_tire_sequence": right["tire_sequence"],
        "left_rank": left_rank,
        "right_rank": right_rank,
        "lgbm_gap": float(left["pred_lgbm"] - right["pred_lgbm"]),
        "abs_lgbm_gap": abs(float(left["pred_lgbm"] - right["pred_lgbm"])),
        "blend_gap": float(left["pred_blend"] - right["pred_blend"]),
        "abs_blend_gap": abs(float(left["pred_blend"] - right["pred_blend"])),
        "left_pit_stop_count": left["pit_stop_count"],
        "right_pit_stop_count": right["pit_stop_count"],
        "pit_stop_gap": left["pit_stop_count"] - right["pit_stop_count"],
        "left_first_stop_lap": left["first_stop_lap"],
        "right_first_stop_lap": right["first_stop_lap"],
        "first_stop_gap": left["first_stop_lap"] - right["first_stop_lap"],
        "track_temp": left["track_temp"],
        "total_laps": left["total_laps"],
        "pit_lane_time": left["pit_lane_time"],
    }


def rerank_front(frame, pair_model):
    reranked = []
    for _, race in frame.groupby("race_id", sort=False):
        ranked = race.sort_values(["pred_lgbm", "driver_id"], ascending=[True, True]).reset_index(drop=True)
        head = ranked.head(TOP_K).copy()
        tail = ranked.iloc[TOP_K:].copy()
        adjustment = np.zeros(len(head), dtype=float)

        pairs = []
        positions = []
        for i, j in itertools.combinations(range(len(head)), 2):
            left = head.iloc[i]
            right = head.iloc[j]
            if abs(float(left["pred_lgbm"] - right["pred_lgbm"])) > FRONT_GAP:
                continue
            pairs.append(pair_features(left, right, i + 1, j + 1))
            positions.append((i, j))

        if pairs:
            probs = pair_model.predict_proba(pd.DataFrame(pairs))[:, 1]
            for (i, j), prob_left_ahead in zip(positions, probs):
                centered = prob_left_ahead - 0.5
                adjustment[i] -= 0.25 * centered
                adjustment[j] += 0.25 * centered

        head["pred_rerank"] = head["pred_lgbm"] + adjustment
        tail["pred_rerank"] = tail["pred_lgbm"]
        combined = pd.concat([head, tail], axis=0, ignore_index=True)
        reranked.append(combined)

    return pd.concat(reranked, axis=0, ignore_index=True)


def evaluate(frame, score_column):
    scored = frame[["race_id", "driver_id", "finish_rank", score_column]].copy()

    exact = 0
    top3 = 0
    top5 = 0
    kendalls = []
    spearmans = []

    for _, race in scored.groupby("race_id", sort=False):
        predicted = race.sort_values([score_column, "driver_id"], ascending=[True, True])
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
    df, splits, v2_record, v4_record, metadata, booster = load_inputs()
    df = add_physics_predictions(df, v2_record, v4_record)
    df = add_lgbm_scores(df, metadata, booster)

    train = df[df["race_id"].isin(splits["train"])].copy()
    validation = df[df["race_id"].isin(splits["validation"])].copy()
    test = df[df["race_id"].isin(splits["test"])].copy()

    sampled_train_races = train["race_id"].drop_duplicates().sample(n=min(TRAIN_RACES, train["race_id"].nunique()), random_state=42)
    pair_train = build_pair_dataset(train[train["race_id"].isin(sampled_train_races)].copy())

    pair_model = build_pair_model()
    feature_cols = PAIR_CATEGORICAL + PAIR_NUMERICAL
    pair_model.fit(pair_train[feature_cols], pair_train["y"])

    validation_reranked = rerank_front(validation, pair_model)
    test_reranked = rerank_front(test, pair_model)

    metrics = {
        "pair_dataset": {
            "rows": int(len(pair_train)),
            "baseline_pair_accuracy": float((pair_train["baseline_left_ahead"] == pair_train["y"]).mean()),
        },
        "validation": {
            "lgbm_ranker": evaluate(validation.assign(pred_rerank=validation["pred_lgbm"]), "pred_rerank"),
            "topk_reranked": evaluate(validation_reranked, "pred_rerank"),
        },
        "test": {
            "lgbm_ranker": evaluate(test.assign(pred_rerank=test["pred_lgbm"]), "pred_rerank"),
            "topk_reranked": evaluate(test_reranked, "pred_rerank"),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
