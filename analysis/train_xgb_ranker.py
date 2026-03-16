#!/usr/bin/env python3
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from xgboost import XGBRanker

import calibrate_physics_simulator_v2 as sim_v2
import calibrate_physics_simulator_v4 as sim_v4


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "analysis" / "historical_driver_features.csv"
SPLIT_PATH = ROOT / "analysis" / "splits" / "race_id_splits.json"
V2_PARAMS_PATH = ROOT / "analysis" / "models" / "physics_simulator_v2" / "best_params.json"
V4_PARAMS_PATH = ROOT / "analysis" / "models" / "physics_simulator_v4" / "best_params.json"
OUTPUT_DIR = ROOT / "analysis" / "models" / "xgb_ranker"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
MODEL_PATH = OUTPUT_DIR / "model.json"
METADATA_PATH = OUTPUT_DIR / "metadata.json"

TRAIN_RACES = int(os.environ.get("TRAIN_RACES", "600"))
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", "300"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.05"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "6"))
USE_COMPOSITION_FEATURES = os.environ.get("USE_COMPOSITION_FEATURES", "1") != "0"


BASE_FEATURE_COLUMNS = [
    "track",
    "starting_tire",
    "tire_sequence",
    "stint_1_compound",
    "stint_2_compound",
    "stint_3_compound",
    "temp_band",
    "lap_band",
    "sequence_temp_band",
    "sequence_lap_band",
    "start_tire_lap_band",
    "start_tire_temp_band",
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
    "first_stop_frac",
    "second_stop_frac",
    "soft_laps",
    "medium_laps",
    "hard_laps",
    "soft_frac",
    "medium_frac",
    "hard_frac",
    "soft_max_stint",
    "medium_max_stint",
    "hard_max_stint",
    "stint_1_laps",
    "stint_2_laps",
    "stint_3_laps",
    "stint_1_frac",
    "stint_2_frac",
    "stint_3_frac",
]

BASE_CATEGORICAL_COLUMNS = [
    "track",
    "starting_tire",
    "tire_sequence",
    "stint_1_compound",
    "stint_2_compound",
    "stint_3_compound",
    "temp_band",
    "lap_band",
    "sequence_temp_band",
    "sequence_lap_band",
    "start_tire_lap_band",
    "start_tire_temp_band",
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


def add_regime_features(df):
    out = df.copy()
    out["temp_band"] = pd.cut(
        out["track_temp"],
        bins=[17, 24, 30, 36, 42],
        labels=["cool", "mild", "warm", "hot"],
        include_lowest=True,
    ).astype(str)
    out["lap_band"] = pd.cut(
        out["total_laps"],
        bins=[24, 35, 45, 55, 70],
        labels=["short", "mid_short", "mid", "long"],
        include_lowest=True,
    ).astype(str)
    out["first_stop_frac"] = out["first_stop_lap"].fillna(0.0) / out["total_laps"]
    out["second_stop_frac"] = out["second_stop_lap"].fillna(0.0) / out["total_laps"]
    out["soft_frac"] = out["soft_laps"] / out["total_laps"]
    out["medium_frac"] = out["medium_laps"] / out["total_laps"]
    out["hard_frac"] = out["hard_laps"] / out["total_laps"]
    out["stint_1_frac"] = out["stint_1_laps"] / out["total_laps"]
    out["stint_2_frac"] = out["stint_2_laps"] / out["total_laps"]
    out["stint_3_frac"] = out["stint_3_laps"] / out["total_laps"]
    out["sequence_temp_band"] = out["tire_sequence"].astype(str) + "|" + out["temp_band"]
    out["sequence_lap_band"] = out["tire_sequence"].astype(str) + "|" + out["lap_band"]
    out["start_tire_lap_band"] = out["starting_tire"].astype(str) + "|" + out["lap_band"]
    out["start_tire_temp_band"] = out["starting_tire"].astype(str) + "|" + out["temp_band"]
    return out


def derive_major_sequences(train_df, top_n=8):
    return train_df["tire_sequence"].value_counts().head(top_n).index.tolist()


def add_race_composition_features(df, major_sequences):
    out = df.copy()
    race_size = out.groupby("race_id")["driver_id"].transform("size").astype(float)
    out["race_size"] = race_size
    out["unique_sequence_count"] = out.groupby("race_id")["tire_sequence"].transform("nunique").astype(float)
    out["sequence_field_count"] = out.groupby(["race_id", "tire_sequence"])["driver_id"].transform("size").astype(float)
    out["sequence_field_share"] = out["sequence_field_count"] / race_size
    out["starting_tire_field_count"] = out.groupby(["race_id", "starting_tire"])["driver_id"].transform("size").astype(float)
    out["starting_tire_field_share"] = out["starting_tire_field_count"] / race_size
    out["one_stop_field_count"] = out.groupby("race_id")["pit_stop_count"].transform(lambda s: float((s == 1).sum()))
    out["two_stop_field_count"] = out.groupby("race_id")["pit_stop_count"].transform(lambda s: float((s == 2).sum()))
    out["one_stop_field_share"] = out["one_stop_field_count"] / race_size
    out["two_stop_field_share"] = out["two_stop_field_count"] / race_size

    for seq in major_sequences:
        safe = seq.lower().replace(">", "_to_")
        count_col = f"field_count__{safe}"
        share_col = f"field_share__{safe}"
        seq_counts = out.groupby("race_id")["tire_sequence"].transform(lambda s, target=seq: float((s == target).sum()))
        out[count_col] = seq_counts
        out[share_col] = seq_counts / race_size

    return out


def get_feature_config(major_sequences):
    feature_columns = list(BASE_FEATURE_COLUMNS) + [
        "race_size",
        "unique_sequence_count",
        "sequence_field_count",
        "sequence_field_share",
        "starting_tire_field_count",
        "starting_tire_field_share",
        "one_stop_field_count",
        "two_stop_field_count",
        "one_stop_field_share",
        "two_stop_field_share",
    ]
    for seq in major_sequences:
        safe = seq.lower().replace(">", "_to_")
        feature_columns.append(f"field_count__{safe}")
        feature_columns.append(f"field_share__{safe}")
    categorical_columns = list(BASE_CATEGORICAL_COLUMNS)
    return feature_columns, categorical_columns


def one_hot_encode(train, validation, test, categorical_columns):
    combined = pd.concat([train, validation, test], axis=0, ignore_index=True)
    categories = {}
    for col in categorical_columns:
        categories[col] = sorted(combined[col].fillna("__MISSING__").astype(str).unique().tolist())
    combined = pd.get_dummies(
        combined,
        columns=categorical_columns,
        dummy_na=True,
    )

    train_rows = len(train)
    validation_rows = len(validation)
    train_enc = combined.iloc[:train_rows].copy()
    validation_enc = combined.iloc[train_rows : train_rows + validation_rows].copy()
    test_enc = combined.iloc[train_rows + validation_rows :].copy()
    return train_enc, validation_enc, test_enc, categories, combined.columns.tolist()


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

    ranker = XGBRanker(
        objective="rank:pairwise",
        tree_method="hist",
        learning_rate=LEARNING_RATE,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=1,
        random_state=42,
    )

    ranker.fit(
        train_x,
        train["finish_rank"],
        group=group_sizes(train),
        eval_set=[(validation_x, validation["finish_rank"])],
        eval_group=[group_sizes(validation)],
        verbose=False,
    )

    metrics = {
        "validation": {
            "physics_blend": evaluate(validation, validation["pred_blend"].to_numpy()),
            "xgb_ranker": evaluate(validation, ranker.predict(validation_x)),
        },
        "test": {
            "physics_blend": evaluate(test, test["pred_blend"].to_numpy()),
            "xgb_ranker": evaluate(test, ranker.predict(test_x)),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    ranker.save_model(str(MODEL_PATH))
    METADATA_PATH.write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "categorical_columns": categorical_columns,
                "categories": categories,
                "encoded_columns": encoded_columns,
                "major_sequences": major_sequences,
                "use_composition_features": USE_COMPOSITION_FEATURES,
                "train_races": len(sampled_train_races),
                "n_estimators": N_ESTIMATORS,
                "learning_rate": LEARNING_RATE,
                "max_depth": MAX_DEPTH,
            },
            indent=2,
        )
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
