#!/usr/bin/env python3
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "analysis" / "models" / "full_pairwise_model"
MODEL_PATH = MODEL_DIR / "model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

SOLUTION_DIR = ROOT / "solution"
if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

import race_simulator_xgb_gate as gate  # noqa: E402


PAIR_CATEGORICAL = [
    "branch",
    "track",
    "temp_band",
    "lap_band",
    "top_sequence",
    "top_count_bucket",
    "left_starting_tire",
    "right_starting_tire",
    "left_tire_sequence",
    "right_tire_sequence",
]

PAIR_NUMERICAL = [
    "gate_score_gap",
    "abs_gate_score_gap",
    "gate_rank_gap",
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
    "unique_sequence_count",
    "one_stop_field_share",
    "two_stop_field_share",
]


def top_count_bucket(top_count):
    if top_count <= 5:
        return "4_5"
    if top_count <= 7:
        return "6_7"
    return "8_10"


def enrich_race_with_gate(frame):
    race = frame.copy()
    branch, metadata, model = gate.choose_bundle(race)
    if metadata.get("use_composition_features", False):
        race = gate.add_race_composition_features(race, metadata)
    encoded = gate.encode(race, metadata)
    race["pred_score"] = model.predict(encoded)
    if branch == "comp":
        race = gate.apply_comp_tiebreak(race)
        race = gate.apply_comp_tail_tiebreak(race)
        race = gate.apply_comp_template_tail_rules(race)
        race = gate.apply_reverse_engineered_tail_rules(race, "comp")
    elif branch == "balanced":
        race = gate.apply_balanced_template_tail_rules(race)
        race = gate.apply_reverse_engineered_tail_rules(race, "balanced")
    else:
        race = gate.apply_reverse_engineered_tail_rules(race, "no_comp")
    race = race.sort_values(["pred_score", "driver_id"], ascending=[True, True]).reset_index(drop=True)
    race["gate_rank"] = np.arange(1, len(race) + 1, dtype=float)
    sequence_counts = race["tire_sequence"].value_counts()
    race["branch"] = branch
    race["top_sequence"] = str(sequence_counts.index[0])
    race["top_count_bucket"] = top_count_bucket(int(sequence_counts.iloc[0]))
    return race


def pair_feature_row(left, right):
    return {
        "branch": str(left["branch"]),
        "track": str(left["track"]),
        "temp_band": str(left["temp_band"]),
        "lap_band": str(left["lap_band"]),
        "top_sequence": str(left["top_sequence"]),
        "top_count_bucket": str(left["top_count_bucket"]),
        "left_starting_tire": str(left["starting_tire"]),
        "right_starting_tire": str(right["starting_tire"]),
        "left_tire_sequence": str(left["tire_sequence"]),
        "right_tire_sequence": str(right["tire_sequence"]),
        "gate_score_gap": float(left["pred_score"] - right["pred_score"]),
        "abs_gate_score_gap": abs(float(left["pred_score"] - right["pred_score"])),
        "gate_rank_gap": float(left["gate_rank"] - right["gate_rank"]),
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
        "unique_sequence_count": float(left["unique_sequence_count"]),
        "one_stop_field_share": float(left["one_stop_field_share"]),
        "two_stop_field_share": float(left["two_stop_field_share"]),
    }


def load_model_bundle():
    with MODEL_PATH.open("rb") as fh:
        model = pickle.load(fh)
    metadata = json.loads(METADATA_PATH.read_text())
    return model, metadata


def pairwise_expected_wins(race, model):
    n = len(race)
    expected_wins = np.zeros(n, dtype=float)
    pair_rows = []
    pair_positions = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            pair_rows.append(pair_feature_row(race.iloc[i], race.iloc[j]))
            pair_positions.append((i, j))
    pair_frame = pd.DataFrame(pair_rows)
    probs = model.predict_proba(pair_frame)[:, 1]
    for (i, j), prob_left_ahead in zip(pair_positions, probs):
        expected_wins[i] += float(prob_left_ahead)
        expected_wins[j] += 1.0 - float(prob_left_ahead)
    return expected_wins


def rank_race_with_pairwise_model(race, model):
    ranked = enrich_race_with_gate(race)
    ranked["expected_wins"] = pairwise_expected_wins(ranked, model)
    ranked = ranked.sort_values(
        ["expected_wins", "pred_score", "driver_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return ranked
