#!/usr/bin/env python3
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from xgboost import XGBRanker

import calibrate_physics_simulator_v2 as sim_v2
import calibrate_physics_simulator_v4 as sim_v4


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "analysis" / "historical_driver_features.csv"
V2_PARAMS_PATH = ROOT / "analysis" / "models" / "physics_simulator_v2" / "best_params.json"
V4_PARAMS_PATH = ROOT / "analysis" / "models" / "physics_simulator_v4" / "best_params.json"
OUTPUT_DIR = ROOT / "analysis" / "hard_split_eval"
RESULTS_PATH = OUTPUT_DIR / "results.json"


FEATURE_COLUMNS = [
    "track",
    "starting_tire",
    "tire_sequence",
    "stint_1_compound",
    "stint_2_compound",
    "stint_3_compound",
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

CATEGORICAL_COLUMNS = [
    "track",
    "starting_tire",
    "tire_sequence",
    "stint_1_compound",
    "stint_2_compound",
    "stint_3_compound",
]

NUMERIC_COLUMNS = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_COLUMNS]


def load_inputs():
    df = pd.read_csv(DATA_PATH)
    v2 = json.loads(V2_PARAMS_PATH.read_text())
    v4 = json.loads(V4_PARAMS_PATH.read_text())
    return df, v2, v4


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


def evaluate(frame, pred_score):
    scored = frame[["race_id", "driver_id", "finish_rank"]].copy()
    scored["pred_score"] = pred_score

    exact = 0
    kendalls = []
    spearmans = []
    for _, race in scored.groupby("race_id", sort=False):
        predicted = race.sort_values(["pred_score", "driver_id"], ascending=[True, True])
        actual = race.sort_values(["finish_rank", "driver_id"], ascending=[True, True])
        pred_ids = predicted["driver_id"].tolist()
        actual_ids = actual["driver_id"].tolist()
        exact += int(pred_ids == actual_ids)

        merged = actual[["driver_id", "finish_rank"]].merge(
            predicted[["driver_id"]].assign(pred_rank=np.arange(1, len(predicted) + 1)),
            on="driver_id",
            how="inner",
        )
        kendalls.append(float(kendalltau(merged["finish_rank"], merged["pred_rank"]).statistic))
        spearmans.append(float(spearmanr(merged["finish_rank"], merged["pred_rank"]).statistic))

    races = frame["race_id"].nunique()
    return {
        "races": int(races),
        "exact_order_accuracy": exact / races,
        "mean_kendall_tau": float(np.mean(kendalls)),
        "mean_spearman_rho": float(np.mean(spearmans)),
    }


def encode_lgbm(train, test):
    train = train.copy()
    test = test.copy()
    categories = {}
    for col in CATEGORICAL_COLUMNS:
        train[col] = train[col].fillna("__MISSING__").astype("category")
        categories[col] = train[col].cat.categories.tolist()
        test[col] = pd.Categorical(test[col].fillna("__MISSING__"), categories=categories[col])
    for col in NUMERIC_COLUMNS:
        median = float(train[col].dropna().median()) if not train[col].dropna().empty else 0.0
        train[col] = train[col].fillna(median)
        test[col] = test[col].fillna(median)
    return train, test


def encode_xgb(train, test):
    train_feat = train[FEATURE_COLUMNS].copy()
    test_feat = test[FEATURE_COLUMNS].copy()
    combined = pd.concat([train_feat, test_feat], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=CATEGORICAL_COLUMNS, dummy_na=True)
    train_rows = len(train_feat)
    train_x = combined.iloc[:train_rows].copy()
    test_x = combined.iloc[train_rows:].copy()
    return train_x, test_x


def fit_lgbm(train, test):
    train_enc, test_enc = encode_lgbm(train, test)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        learning_rate=0.03,
        n_estimators=400,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=1,
        force_row_wise=True,
        random_state=42,
    )
    model.fit(
        train_enc[FEATURE_COLUMNS],
        train_enc["finish_rank"],
        group=train_enc.groupby("race_id", sort=False).size().tolist(),
        categorical_feature=CATEGORICAL_COLUMNS,
    )
    return model.predict(test_enc[FEATURE_COLUMNS])


def fit_xgb(train, test):
    train_x, test_x = encode_xgb(train, test)
    model = XGBRanker(
        objective="rank:pairwise",
        tree_method="hist",
        learning_rate=0.03,
        n_estimators=400,
        max_depth=8,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=1,
        random_state=42,
    )
    model.fit(
        train_x,
        train["finish_rank"],
        group=train.groupby("race_id", sort=False).size().tolist(),
        verbose=False,
    )
    return model.predict(test_x)


def run_split(df, split_name, train_mask, test_mask):
    train = df[train_mask].copy()
    test = df[test_mask].copy()
    if train["race_id"].nunique() == 0 or test["race_id"].nunique() == 0:
        return None

    return {
        "split_name": split_name,
        "physics_blend": evaluate(test, test["pred_blend"].to_numpy()),
        "lgbm_ranker": evaluate(test, fit_lgbm(train, test)),
        "xgb_ranker": evaluate(test, fit_xgb(train, test)),
    }


def main():
    df, v2_record, v4_record = load_inputs()
    df = add_physics_predictions(df, v2_record, v4_record)

    results = {"leave_one_track_out": [], "temp_band_holdout": [], "lap_band_holdout": []}

    for track in sorted(df["track"].unique()):
        test_races = df["track"] == track
        train_races = ~test_races
        result = run_split(df, f"track={track}", train_races, test_races)
        if result:
            results["leave_one_track_out"].append(result)

    temp_bins = pd.cut(df["track_temp"], bins=[17, 24, 30, 36, 42], labels=["cool", "mild", "warm", "hot"])
    for band in temp_bins.cat.categories:
        test_races = temp_bins == band
        train_races = ~test_races
        result = run_split(df, f"temp_band={band}", train_races, test_races)
        if result:
            results["temp_band_holdout"].append(result)

    lap_bins = pd.cut(df["total_laps"], bins=[24, 35, 45, 55, 70], labels=["short", "mid_short", "mid", "long"])
    for band in lap_bins.cat.categories:
        test_races = lap_bins == band
        train_races = ~test_races
        result = run_split(df, f"lap_band={band}", train_races, test_races)
        if result:
            results["lap_band_holdout"].append(result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
