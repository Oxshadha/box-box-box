#!/usr/bin/env python3
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from train_xgb_ranker import (
    add_physics_predictions,
    add_regime_features,
    load_inputs,
)


ROOT = Path(__file__).resolve().parent.parent

RACE_NUMERIC = [
    "total_laps",
    "track_temp",
    "pit_lane_time",
    "base_lap_time",
    "unique_sequence_count",
    "two_stop_count",
    "top_sequence_count",
    "soft_share",
    "medium_share",
    "hard_share",
]
RACE_CATEGORICAL = ["track", "temp_band", "lap_band", "top_sequence"]
DRIVER_NUMERIC = [
    "pit_stop_count",
    "first_stop_frac",
    "second_stop_frac",
    "soft_frac",
    "medium_frac",
    "hard_frac",
    "stint_1_frac",
    "stint_2_frac",
    "stint_3_frac",
    "pred_blend_rel",
    "pred_v2_rel",
    "pred_v4_rel",
]
DRIVER_CATEGORICAL = ["starting_tire", "tire_sequence", "stint_1_compound", "stint_2_compound", "stint_3_compound"]


def add_full_features(df):
    df = add_regime_features(df.copy())
    race_mean = df.groupby("race_id")["pred_blend"].transform("mean")
    race_std = df.groupby("race_id")["pred_blend"].transform("std").replace(0, 1).fillna(1)
    df["pred_blend_rel"] = (df["pred_blend"] - race_mean) / race_std
    race_mean_v2 = df.groupby("race_id")["pred_v2"].transform("mean")
    race_std_v2 = df.groupby("race_id")["pred_v2"].transform("std").replace(0, 1).fillna(1)
    df["pred_v2_rel"] = (df["pred_v2"] - race_mean_v2) / race_std_v2
    race_mean_v4 = df.groupby("race_id")["pred_v4"].transform("mean")
    race_std_v4 = df.groupby("race_id")["pred_v4"].transform("std").replace(0, 1).fillna(1)
    df["pred_v4_rel"] = (df["pred_v4"] - race_mean_v4) / race_std_v4
    return df


def summarize_races(df):
    rows = []
    for race_id, race in df.groupby("race_id", sort=False):
        seq_counts = race["tire_sequence"].value_counts()
        top_sequence = str(seq_counts.index[0])
        top_sequence_count = int(seq_counts.iloc[0])
        total = float(len(race))
        rows.append(
            {
                "race_id": race_id,
                "track": str(race["track"].iloc[0]),
                "temp_band": str(race["temp_band"].iloc[0]),
                "lap_band": str(race["lap_band"].iloc[0]),
                "top_sequence": top_sequence,
                "total_laps": float(race["total_laps"].iloc[0]),
                "track_temp": float(race["track_temp"].iloc[0]),
                "pit_lane_time": float(race["pit_lane_time"].iloc[0]),
                "base_lap_time": float(race["base_lap_time"].iloc[0]),
                "unique_sequence_count": float(race["tire_sequence"].nunique()),
                "two_stop_count": float((race["pit_stop_count"] == 2).sum()),
                "top_sequence_count": float(top_sequence_count),
                "soft_share": float((race["starting_tire"] == "SOFT").sum()) / total,
                "medium_share": float((race["starting_tire"] == "MEDIUM").sum()) / total,
                "hard_share": float((race["starting_tire"] == "HARD").sum()) / total,
            }
        )
    return pd.DataFrame(rows)


def fit_race_encoder(race_summary):
    categories = {}
    for col in RACE_CATEGORICAL:
        categories[col] = sorted(race_summary[col].fillna("__MISSING__").astype(str).unique().tolist())
    means = {col: float(race_summary[col].mean()) for col in RACE_NUMERIC}
    stds = {col: float(race_summary[col].std()) if float(race_summary[col].std()) > 0 else 1.0 for col in RACE_NUMERIC}
    return {"categories": categories, "means": means, "stds": stds}


def encode_races(race_summary, encoder):
    frame = race_summary.copy()
    parts = []
    for col in RACE_NUMERIC:
        vals = (frame[col].astype(float) - encoder["means"][col]) / encoder["stds"][col]
        parts.append(vals.to_numpy().reshape(-1, 1))
    for col in RACE_CATEGORICAL:
        cur = frame[col].fillna("__MISSING__").astype(str)
        for cat in encoder["categories"][col]:
            parts.append((cur == cat).astype(float).to_numpy().reshape(-1, 1))
    return np.hstack(parts) if parts else np.empty((len(frame), 0))


def driver_distance_matrix(target_race, source_race):
    left = target_race.reset_index(drop=True).copy()
    right = source_race.reset_index(drop=True).copy()
    cost = np.zeros((len(left), len(right)), dtype=float)
    for col in DRIVER_NUMERIC:
        lvals = left[col].astype(float).to_numpy()[:, None]
        rvals = right[col].astype(float).to_numpy()[None, :]
        cost += np.abs(lvals - rvals)
    for col in DRIVER_CATEGORICAL:
        lvals = left[col].fillna("__MISSING__").astype(str).to_numpy()[:, None]
        rvals = right[col].fillna("__MISSING__").astype(str).to_numpy()[None, :]
        cost += (lvals != rvals).astype(float) * 1.5
    return cost


def map_order_from_source(target_race, source_race):
    target = target_race.reset_index(drop=True).copy()
    source = source_race.reset_index(drop=True).copy()
    cost = driver_distance_matrix(target, source)
    row_ind, col_ind = linear_sum_assignment(cost)
    mapped = pd.DataFrame(
        {
            "driver_id": target.iloc[row_ind]["driver_id"].to_numpy(),
            "mapped_finish_rank": source.iloc[col_ind]["finish_rank"].astype(int).to_numpy(),
        }
    )
    mapped = mapped.sort_values(["mapped_finish_rank", "driver_id"], ascending=[True, True])
    total_cost = float(cost[row_ind, col_ind].sum())
    return mapped["driver_id"].tolist(), total_cost


def build_artifact(df, train_race_ids, top_k=7):
    train_df = df[df["race_id"].isin(train_race_ids)].copy()
    race_summary = summarize_races(train_df)
    encoder = fit_race_encoder(race_summary)
    race_matrix = encode_races(race_summary, encoder)
    race_records = []
    for race_id, race in train_df.groupby("race_id", sort=False):
        ordered = race.sort_values(["finish_rank", "driver_id"], ascending=[True, True]).copy()
        race_records.append(
            {
                "race_id": race_id,
                "rows": ordered[
                    [
                        "driver_id",
                        "finish_rank",
                        "track",
                        "total_laps",
                        "track_temp",
                        "pit_lane_time",
                        "base_lap_time",
                        "starting_tire",
                        "pit_stop_count",
                        "first_stop_frac",
                        "second_stop_frac",
                        "tire_sequence",
                        "stint_1_compound",
                        "stint_2_compound",
                        "stint_3_compound",
                        "soft_frac",
                        "medium_frac",
                        "hard_frac",
                        "stint_1_frac",
                        "stint_2_frac",
                        "stint_3_frac",
                        "pred_blend_rel",
                        "pred_v2_rel",
                        "pred_v4_rel",
                    ]
                ].to_dict(orient="records"),
            }
        )
    return {
        "top_k": top_k,
        "race_summary_rows": race_summary.to_dict(orient="records"),
        "race_encoder": encoder,
        "race_matrix": race_matrix,
        "train_races": race_records,
    }


def prepare_test_race_frame(test_case):
    rows = []
    for grid_slot in sorted(test_case["strategies"]):
        strategy = test_case["strategies"][grid_slot]
        total_laps = float(test_case["race_config"]["total_laps"])
        first_stop_lap = float(strategy["pit_stops"][0]["lap"] if strategy["pit_stops"] else 0.0)
        second_stop_lap = float(strategy["pit_stops"][1]["lap"] if len(strategy["pit_stops"]) > 1 else 0.0)
        tire_sequence = ">".join([strategy["starting_tire"]] + [stop["to_tire"] for stop in strategy["pit_stops"]])
        stints = []
        tire = strategy["starting_tire"]
        prev = 0
        for stop in strategy["pit_stops"]:
            stints.append((tire, stop["lap"] - prev))
            tire = stop["to_tire"]
            prev = stop["lap"]
        stints.append((tire, total_laps - prev))
        row = {
            "driver_id": strategy["driver_id"],
            "track": test_case["race_config"]["track"],
            "total_laps": total_laps,
            "track_temp": float(test_case["race_config"]["track_temp"]),
            "pit_lane_time": float(test_case["race_config"]["pit_lane_time"]),
            "base_lap_time": float(test_case["race_config"]["base_lap_time"]),
            "starting_tire": strategy["starting_tire"],
            "pit_stop_count": len(strategy["pit_stops"]),
            "first_stop_frac": first_stop_lap / total_laps,
            "second_stop_frac": second_stop_lap / total_laps,
            "tire_sequence": tire_sequence,
            "stint_1_compound": stints[0][0] if len(stints) > 0 else None,
            "stint_2_compound": stints[1][0] if len(stints) > 1 else None,
            "stint_3_compound": stints[2][0] if len(stints) > 2 else None,
            "soft_frac": 0.0,
            "medium_frac": 0.0,
            "hard_frac": 0.0,
            "stint_1_frac": float(stints[0][1] if len(stints) > 0 else 0) / total_laps,
            "stint_2_frac": float(stints[1][1] if len(stints) > 1 else 0) / total_laps,
            "stint_3_frac": float(stints[2][1] if len(stints) > 2 else 0) / total_laps,
            "pred_blend_rel": 0.0,
            "pred_v2_rel": 0.0,
            "pred_v4_rel": 0.0,
        }
        for compound, laps in stints:
            row[f"{compound.lower()}_frac"] += float(laps) / total_laps
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame["track_temp"].iloc[0] <= 24:
        frame["temp_band"] = "cool"
    elif frame["track_temp"].iloc[0] <= 30:
        frame["temp_band"] = "mild"
    elif frame["track_temp"].iloc[0] <= 36:
        frame["temp_band"] = "warm"
    else:
        frame["temp_band"] = "hot"
    if frame["total_laps"].iloc[0] <= 35:
        frame["lap_band"] = "short"
    elif frame["total_laps"].iloc[0] <= 45:
        frame["lap_band"] = "mid_short"
    elif frame["total_laps"].iloc[0] <= 55:
        frame["lap_band"] = "mid"
    else:
        frame["lap_band"] = "long"
    seq_counts = frame["tire_sequence"].value_counts()
    top_sequence = str(seq_counts.index[0])
    top_sequence_count = float(seq_counts.iloc[0])
    race_size = float(len(frame))
    summary = pd.DataFrame(
        [
            {
                "race_id": test_case["race_id"],
                "track": str(frame["track"].iloc[0]),
                "temp_band": str(frame["temp_band"].iloc[0]),
                "lap_band": str(frame["lap_band"].iloc[0]),
                "top_sequence": top_sequence,
                "total_laps": float(frame["total_laps"].iloc[0]),
                "track_temp": float(frame["track_temp"].iloc[0]),
                "pit_lane_time": float(frame["pit_lane_time"].iloc[0]),
                "base_lap_time": float(frame["base_lap_time"].iloc[0]),
                "unique_sequence_count": float(frame["tire_sequence"].nunique()),
                "two_stop_count": float((frame["pit_stop_count"] == 2).sum()),
                "top_sequence_count": top_sequence_count,
                "soft_share": float((frame["starting_tire"] == "SOFT").sum()) / race_size,
                "medium_share": float((frame["starting_tire"] == "MEDIUM").sum()) / race_size,
                "hard_share": float((frame["starting_tire"] == "HARD").sum()) / race_size,
            }
        ]
    )
    return frame, summary


def nearest_order(test_case, artifact):
    frame, summary = prepare_test_race_frame(test_case)
    race_vector = encode_races(summary, artifact["race_encoder"])
    train_matrix = artifact["race_matrix"]
    distances = ((train_matrix - race_vector[0]) ** 2).sum(axis=1)
    ranked_idx = np.argsort(distances)[: int(artifact.get("top_k", 7))]

    best_order = None
    best_cost = None
    for idx in ranked_idx:
        source = pd.DataFrame(artifact["train_races"][int(idx)]["rows"])
        order, cost = map_order_from_source(frame, source)
        score = float(distances[int(idx)]) + cost / max(len(frame), 1)
        if best_cost is None or score < best_cost:
            best_cost = score
            best_order = order
    return best_order


def load_artifact(model_dir):
    with (Path(model_dir) / "artifact.pkl").open("rb") as fh:
        return pickle.load(fh)


def save_artifact(model_dir, artifact, metrics):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "artifact.pkl").open("wb") as fh:
        pickle.dump(artifact, fh)
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def load_dataset():
    df, splits, v2_record, v4_record = load_inputs()
    df = add_physics_predictions(df, v2_record, v4_record)
    df = add_full_features(df)
    return df, splits
