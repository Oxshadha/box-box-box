#!/usr/bin/env python3
import itertools
import json
from pathlib import Path

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
OUTPUT_DIR = ROOT / "analysis" / "models" / "pairwise_residual"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
MODEL_INFO_PATH = OUTPUT_DIR / "model_info.json"

PAIR_GAP = 2.5
PAIR_SAMPLE_LIMIT = 20000
CORRECTION_STRENGTH = 0.35


PAIR_CATEGORICAL = [
    "left_starting_tire",
    "right_starting_tire",
    "left_tire_sequence",
    "right_tire_sequence",
    "track",
]

PAIR_NUMERICAL = [
    "blend_gap",
    "abs_blend_gap",
    "left_pit_stop_count",
    "right_pit_stop_count",
    "pit_stop_count_gap",
    "left_first_stop_lap",
    "right_first_stop_lap",
    "first_stop_lap_gap",
    "track_temp",
    "total_laps",
    "pit_lane_time",
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


def build_pairwise_dataset(frame: pd.DataFrame, max_pairs: int | None = None):
    rows = []
    for _, race in frame.groupby("race_id", sort=False):
        race_rows = race.to_dict("records")
        for left, right in itertools.combinations(race_rows, 2):
            gap = left["pred_blend"] - right["pred_blend"]
            if abs(gap) > PAIR_GAP:
                continue

            # Baseline predicts lower score ahead.
            baseline_left_ahead = gap < 0
            truth_left_ahead = left["finish_rank"] < right["finish_rank"]

            rows.append(
                {
                    "track": left["track"],
                    "left_starting_tire": left["starting_tire"],
                    "right_starting_tire": right["starting_tire"],
                    "left_tire_sequence": left["tire_sequence"],
                    "right_tire_sequence": right["tire_sequence"],
                    "blend_gap": gap,
                    "abs_blend_gap": abs(gap),
                    "left_pit_stop_count": left["pit_stop_count"],
                    "right_pit_stop_count": right["pit_stop_count"],
                    "pit_stop_count_gap": left["pit_stop_count"] - right["pit_stop_count"],
                    "left_first_stop_lap": left["first_stop_lap"],
                    "right_first_stop_lap": right["first_stop_lap"],
                    "first_stop_lap_gap": left["first_stop_lap"] - right["first_stop_lap"],
                    "track_temp": left["track_temp"],
                    "total_laps": left["total_laps"],
                    "pit_lane_time": left["pit_lane_time"],
                    "y": int(truth_left_ahead),
                    "baseline_left_ahead": int(baseline_left_ahead),
                }
            )

    pair_df = pd.DataFrame(rows)
    if max_pairs is not None and len(pair_df) > max_pairs:
        pair_df = pair_df.sample(n=max_pairs, random_state=42).reset_index(drop=True)
    return pair_df


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


def race_pair_features(race: pd.DataFrame, idx_a: int, idx_b: int):
    left = race.iloc[idx_a]
    right = race.iloc[idx_b]
    gap = float(left["pred_blend"] - right["pred_blend"])
    return {
        "track": left["track"],
        "left_starting_tire": left["starting_tire"],
        "right_starting_tire": right["starting_tire"],
        "left_tire_sequence": left["tire_sequence"],
        "right_tire_sequence": right["tire_sequence"],
        "blend_gap": gap,
        "abs_blend_gap": abs(gap),
        "left_pit_stop_count": left["pit_stop_count"],
        "right_pit_stop_count": right["pit_stop_count"],
        "pit_stop_count_gap": left["pit_stop_count"] - right["pit_stop_count"],
        "left_first_stop_lap": left["first_stop_lap"],
        "right_first_stop_lap": right["first_stop_lap"],
        "first_stop_lap_gap": left["first_stop_lap"] - right["first_stop_lap"],
        "track_temp": left["track_temp"],
        "total_laps": left["total_laps"],
        "pit_lane_time": left["pit_lane_time"],
    }


def corrected_scores(frame: pd.DataFrame, pair_model):
    corrected = []
    for _, race in frame.groupby("race_id", sort=False):
        race = race.sort_values(["pred_blend", "driver_id"]).reset_index(drop=True)
        adjustment = np.zeros(len(race), dtype=float)

        close_pairs = []
        pair_positions = []
        for i, j in itertools.combinations(range(len(race)), 2):
            if abs(race.loc[i, "pred_blend"] - race.loc[j, "pred_blend"]) > PAIR_GAP:
                continue
            close_pairs.append(race_pair_features(race, i, j))
            pair_positions.append((i, j))

        if close_pairs:
            probs = pair_model.predict_proba(pd.DataFrame(close_pairs))[:, 1]
            for (i, j), prob_left_ahead in zip(pair_positions, probs):
                centered = prob_left_ahead - 0.5
                adjustment[i] -= CORRECTION_STRENGTH * centered
                adjustment[j] += CORRECTION_STRENGTH * centered

        race["pred_corrected"] = race["pred_blend"] + adjustment
        corrected.append(race)

    return pd.concat(corrected, axis=0, ignore_index=True)


def evaluate(frame: pd.DataFrame, score_column: str):
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
    df, splits, v2_record, v4_record = load_inputs()
    df = add_physics_predictions(df, v2_record, v4_record)

    train = df[df["race_id"].isin(splits["train"])].copy()
    validation = df[df["race_id"].isin(splits["validation"])].copy()
    test = df[df["race_id"].isin(splits["test"])].copy()

    sampled_train_races = train["race_id"].drop_duplicates().sample(n=400, random_state=42)
    pair_train = build_pairwise_dataset(
        train[train["race_id"].isin(sampled_train_races)].copy(),
        max_pairs=PAIR_SAMPLE_LIMIT,
    )

    pair_model = build_pair_model()
    feature_cols = PAIR_CATEGORICAL + PAIR_NUMERICAL
    pair_model.fit(pair_train[feature_cols], pair_train["y"])

    validation_corrected = corrected_scores(validation, pair_model)
    test_corrected = corrected_scores(test, pair_model)

    metrics = {
        "pair_dataset": {
            "rows": int(len(pair_train)),
            "baseline_pair_accuracy": float((pair_train["baseline_left_ahead"] == pair_train["y"]).mean()),
        },
        "validation": {
            "physics_blend": evaluate(validation.assign(pred_corrected=validation["pred_blend"]), "pred_corrected"),
            "pairwise_corrected": evaluate(validation_corrected, "pred_corrected"),
        },
        "test": {
            "physics_blend": evaluate(test.assign(pred_corrected=test["pred_blend"]), "pred_corrected"),
            "pairwise_corrected": evaluate(test_corrected, "pred_corrected"),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    MODEL_INFO_PATH.write_text(
        json.dumps(
            {
                "pair_gap": PAIR_GAP,
                "pair_sample_limit": PAIR_SAMPLE_LIMIT,
                "correction_strength": CORRECTION_STRENGTH,
            },
            indent=2,
        )
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
