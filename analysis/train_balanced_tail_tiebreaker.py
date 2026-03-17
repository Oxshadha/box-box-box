#!/usr/bin/env python3
import itertools
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRanker

import train_xgb_ranker as base


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "analysis" / "models" / "balanced_tail_tiebreaker"
MODEL_PATH = OUTPUT_DIR / "model.pkl"
METADATA_PATH = OUTPUT_DIR / "metadata.json"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
BALANCED_DIR = ROOT / "analysis" / "models" / "xgb_ranker_balanced"

TOP_KEEP = 5
PAIR_GAP = 0.16
MAX_PAIRS = 80000
ADJUSTMENT_STRENGTH = 0.09

PAIR_CATEGORICAL = [
    "track",
    "temp_band",
    "lap_band",
    "left_starting_tire",
    "right_starting_tire",
    "left_tire_sequence",
    "right_tire_sequence",
]

PAIR_NUMERICAL = [
    "score_gap",
    "abs_score_gap",
    "blend_gap",
    "v2_gap",
    "v4_gap",
    "pit_stop_gap",
    "first_stop_frac_gap",
    "second_stop_frac_gap",
    "soft_frac_gap",
    "medium_frac_gap",
    "hard_frac_gap",
    "stint_1_frac_gap",
    "stint_2_frac_gap",
    "stint_3_frac_gap",
    "sequence_field_share_gap",
    "starting_tire_field_share_gap",
    "track_temp",
    "total_laps",
    "pit_lane_time",
]


def load_balanced_bundle():
    metadata = json.loads((BALANCED_DIR / "metadata.json").read_text())
    ranker = XGBRanker()
    ranker.load_model(str(BALANCED_DIR / "model.json"))
    return metadata, ranker


def encode_with_metadata(frame, metadata):
    frame = frame.copy()
    for col in metadata["categorical_columns"]:
        frame[col] = frame[col].fillna("__MISSING__").astype(str)
    numeric_cols = [c for c in metadata["feature_columns"] if c not in metadata["categorical_columns"]]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame = pd.get_dummies(frame, columns=metadata["categorical_columns"], dummy_na=True)
    for col in metadata["encoded_columns"]:
        if col not in frame.columns:
            frame[col] = 0
    return frame[metadata["encoded_columns"]]


def route_balanced_races(frame):
    race_level = (
        frame.groupby("race_id", sort=False)
        .agg(
            total_laps=("total_laps", "first"),
            track_temp=("track_temp", "first"),
            unique_sequences=("tire_sequence", "nunique"),
            two_stop=("pit_stop_count", lambda s: int((s == 2).sum())),
            top_sequence=("tire_sequence", lambda s: str(s.value_counts().index[0])),
            top_count=("tire_sequence", lambda s: int(s.value_counts().iloc[0])),
        )
        .reset_index()
    )
    balanced_cluster = (
        (race_level["two_stop"] == 0)
        & race_level["unique_sequences"].between(4, 6)
        & race_level["track_temp"].between(28, 36)
        & (race_level["total_laps"] <= 45)
        & race_level["top_count"].between(4, 10)
        & race_level["top_sequence"].isin(["MEDIUM>HARD", "SOFT>HARD", "SOFT>MEDIUM", "MEDIUM>SOFT"])
    )
    balanced_races = set(race_level.loc[balanced_cluster, "race_id"])
    return frame[frame["race_id"].isin(balanced_races)].copy()


def pair_row(left, right):
    return {
        "track": left["track"],
        "temp_band": left["temp_band"],
        "lap_band": left["lap_band"],
        "left_starting_tire": left["starting_tire"],
        "right_starting_tire": right["starting_tire"],
        "left_tire_sequence": left["tire_sequence"],
        "right_tire_sequence": right["tire_sequence"],
        "score_gap": float(left["base_score"] - right["base_score"]),
        "abs_score_gap": abs(float(left["base_score"] - right["base_score"])),
        "blend_gap": float(left["pred_blend"] - right["pred_blend"]),
        "v2_gap": float(left["pred_v2"] - right["pred_v2"]),
        "v4_gap": float(left["pred_v4"] - right["pred_v4"]),
        "pit_stop_gap": float(left["pit_stop_count"] - right["pit_stop_count"]),
        "first_stop_frac_gap": float(left["first_stop_frac"] - right["first_stop_frac"]),
        "second_stop_frac_gap": float(left["second_stop_frac"] - right["second_stop_frac"]),
        "soft_frac_gap": float(left["soft_frac"] - right["soft_frac"]),
        "medium_frac_gap": float(left["medium_frac"] - right["medium_frac"]),
        "hard_frac_gap": float(left["hard_frac"] - right["hard_frac"]),
        "stint_1_frac_gap": float(left["stint_1_frac"] - right["stint_1_frac"]),
        "stint_2_frac_gap": float(left["stint_2_frac"] - right["stint_2_frac"]),
        "stint_3_frac_gap": float(left["stint_3_frac"] - right["stint_3_frac"]),
        "sequence_field_share_gap": float(left["sequence_field_share"] - right["sequence_field_share"]),
        "starting_tire_field_share_gap": float(left["starting_tire_field_share"] - right["starting_tire_field_share"]),
        "track_temp": float(left["track_temp"]),
        "total_laps": float(left["total_laps"]),
        "pit_lane_time": float(left["pit_lane_time"]),
    }


def build_pair_dataset(frame):
    rows = []
    for _, race in frame.groupby("race_id", sort=False):
        race = race.sort_values(["base_score", "driver_id"], ascending=[True, True]).reset_index(drop=True)
        if len(race) <= TOP_KEEP:
            continue
        tail = race.iloc[TOP_KEEP:].reset_index(drop=True)
        for i in range(len(tail) - 1):
            j = i + 1
            if abs(float(tail.loc[i, "base_score"] - tail.loc[j, "base_score"])) > PAIR_GAP:
                continue
            left = tail.iloc[i]
            right = tail.iloc[j]
            item = pair_row(left, right)
            item["y"] = int(left["finish_rank"] < right["finish_rank"])
            rows.append(item)
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
            ("model", SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=2000, tol=1e-3, random_state=42)),
        ]
    )


def apply_tiebreaker(frame, model):
    corrected = []
    for _, race in frame.groupby("race_id", sort=False):
        race = race.sort_values(["base_score", "driver_id"], ascending=[True, True]).reset_index(drop=True)
        if len(race) <= TOP_KEEP:
            race["corrected_score"] = race["base_score"]
            corrected.append(race)
            continue
        adjustment = [0.0] * len(race)
        tail = race.iloc[TOP_KEEP:].reset_index(drop=True)
        pair_features = []
        pair_positions = []
        for i in range(len(tail) - 1):
            j = i + 1
            if abs(float(tail.loc[i, "base_score"] - tail.loc[j, "base_score"])) > PAIR_GAP:
                continue
            pair_features.append(pair_row(tail.iloc[i], tail.iloc[j]))
            pair_positions.append((i + TOP_KEEP, j + TOP_KEEP))
        if pair_features:
            probs = model.predict_proba(pd.DataFrame(pair_features))[:, 1]
            for (i, j), prob_left_ahead in zip(pair_positions, probs):
                centered = float(prob_left_ahead) - 0.5
                adjustment[i] -= ADJUSTMENT_STRENGTH * centered
                adjustment[j] += ADJUSTMENT_STRENGTH * centered
        race["corrected_score"] = race["base_score"] + adjustment
        corrected.append(race)
    return pd.concat(corrected, axis=0, ignore_index=True)


def evaluate(frame, score_col):
    exact = 0
    top3 = 0
    top5 = 0
    races = 0
    for _, race in frame.groupby("race_id", sort=False):
        pred = race.sort_values([score_col, "driver_id"], ascending=[True, True])["driver_id"].tolist()
        actual = race.sort_values(["finish_rank", "driver_id"], ascending=[True, True])["driver_id"].tolist()
        exact += int(pred == actual)
        top3 += int(set(pred[:3]) == set(actual[:3]))
        top5 += int(set(pred[:5]) == set(actual[:5]))
        races += 1
    return {
        "races": races,
        "exact_order_accuracy": exact / races if races else 0.0,
        "top3_set_accuracy": top3 / races if races else 0.0,
        "top5_set_accuracy": top5 / races if races else 0.0,
    }


def main():
    df, splits, v2_record, v4_record = base.load_inputs()
    df = base.add_physics_predictions(df, v2_record, v4_record)
    df = base.add_regime_features(df)
    balanced_meta, balanced_model = load_balanced_bundle()
    df = base.add_race_composition_features(df, balanced_meta.get("major_sequences", []))
    encoded = encode_with_metadata(df[balanced_meta["feature_columns"]], balanced_meta)
    df["base_score"] = balanced_model.predict(encoded)
    df = route_balanced_races(df)

    train = df[df["race_id"].isin(splits["train"])].copy()
    validation = df[df["race_id"].isin(splits["validation"])].copy()
    test = df[df["race_id"].isin(splits["test"])].copy()

    train_pairs = build_pair_dataset(train)
    validation_pairs = build_pair_dataset(validation)

    model = build_model()
    model.fit(train_pairs[PAIR_CATEGORICAL + PAIR_NUMERICAL], train_pairs["y"])

    validation_corrected = apply_tiebreaker(validation, model)
    test_corrected = apply_tiebreaker(test, model)
    metrics = {
        "validation": {
            "base": evaluate(validation.assign(corrected_score=validation["base_score"]), "base_score"),
            "tiebreak": evaluate(validation_corrected, "corrected_score"),
            "pair_rows": int(len(validation_pairs)),
        },
        "test": {
            "base": evaluate(test.assign(corrected_score=test["base_score"]), "base_score"),
            "tiebreak": evaluate(test_corrected, "corrected_score"),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as fh:
        pickle.dump(model, fh)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "pair_gap": PAIR_GAP,
                "adjustment_strength": ADJUSTMENT_STRENGTH,
                "top_keep": TOP_KEEP,
            },
            indent=2,
        )
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
