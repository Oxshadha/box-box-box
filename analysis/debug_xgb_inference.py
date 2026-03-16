#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd
from xgboost import XGBRanker


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "analysis" / "models" / "xgb_ranker"
MODEL_PATH = MODEL_DIR / "model.json"
METADATA_PATH = MODEL_DIR / "metadata.json"
INPUT_DIR = ROOT / "data" / "test_cases" / "inputs"

BLEND_ALPHA_V2 = 0.6

V2_THRESHOLDS = {
    "SOFT": 4,
    "MEDIUM": 14,
    "HARD": 20,
}
V2_PARAMS = {
    "soft_offset": -0.6110845761949585,
    "medium_offset": 0.19473605811872785,
    "hard_offset": 0.29777243056406405,
    "soft_fresh_linear": 0.0487811175818378,
    "soft_wear_linear": 0.24745073443787763,
    "soft_wear_quadratic": 0.008074578747492586,
    "soft_temp_wear": 0.0012046249797155024,
    "medium_fresh_linear": 0.018015437515822685,
    "medium_wear_linear": 0.09191172339256137,
    "medium_wear_quadratic": 0.006523972427515913,
    "medium_temp_wear": 0.002646687456306525,
    "hard_fresh_linear": 0.0229104645308332,
    "hard_wear_linear": 0.0643414198827331,
    "hard_wear_quadratic": 0.0011361936089238844,
    "hard_temp_wear": 0.0016637543610475045,
}
V4_THRESHOLDS = {
    "SOFT": 5,
    "MEDIUM": 13,
    "HARD": 23,
}
V4_CLIFF_THRESHOLDS = {
    "SOFT": 7,
    "MEDIUM": 15,
    "HARD": 25,
}
V4_PARAMS = {
    "soft_offset": -1.907015749374082,
    "medium_offset": 0.09066531976590986,
    "hard_offset": 1.2823708973750645,
    "soft_fresh_linear": 0.011927641031369424,
    "soft_wear_linear": 0.266869945443076,
    "soft_wear_quadratic": 0.0015150874613793436,
    "soft_temp_wear": 0.004905301195029219,
    "soft_cliff_penalty": 0.3283386290516841,
    "medium_fresh_linear": 0.030884335964175225,
    "medium_wear_linear": 0.1377369097570092,
    "medium_wear_quadratic": 0.005946871421259991,
    "medium_temp_wear": 0.00035257493827885243,
    "medium_cliff_penalty": 0.06430295882223942,
    "hard_fresh_linear": -0.005003052827774277,
    "hard_wear_linear": 0.07045309901929865,
    "hard_wear_quadratic": 0.0037259406416442226,
    "hard_temp_wear": 0.001890035533729196,
    "hard_cliff_penalty": 0.10362442097381286,
}


with METADATA_PATH.open() as fh:
    METADATA = json.load(fh)

RANKER = XGBRanker()
RANKER.load_model(str(MODEL_PATH))


def build_stints(strategy, total_laps):
    tire = strategy["starting_tire"]
    previous_lap = 0
    stints = []
    for stop in strategy["pit_stops"]:
        stop_lap = stop["lap"]
        stints.append((tire, stop_lap - previous_lap))
        tire = stop["to_tire"]
        previous_lap = stop_lap
    stints.append((tire, total_laps - previous_lap))
    return stints


def s1(n):
    return n * (n + 1.0) / 2.0


def s2(n):
    return n * (n + 1.0) * (2.0 * n + 1.0) / 6.0


def seq(strategy):
    return ">".join([strategy["starting_tire"]] + [stop["to_tire"] for stop in strategy["pit_stops"]])


def v2_score(stints, race_config):
    total = race_config["base_lap_time"] * race_config["total_laps"]
    total += race_config["pit_lane_time"] * (len(stints) - 1)
    temp = race_config["track_temp"]
    for compound, laps in stints:
        prefix = compound.lower()
        fresh = min(laps, V2_THRESHOLDS[compound])
        worn = max(laps - V2_THRESHOLDS[compound], 0)
        total += V2_PARAMS[f"{prefix}_offset"] * laps
        total += V2_PARAMS[f"{prefix}_fresh_linear"] * s1(fresh)
        total += V2_PARAMS[f"{prefix}_wear_linear"] * s1(worn)
        total += V2_PARAMS[f"{prefix}_wear_quadratic"] * s2(worn)
        total += V2_PARAMS[f"{prefix}_temp_wear"] * temp * s1(worn)
    return total


def v4_score(stints, race_config):
    total = race_config["base_lap_time"] * race_config["total_laps"]
    total += race_config["pit_lane_time"] * (len(stints) - 1)
    temp = race_config["track_temp"]
    for compound, laps in stints:
        prefix = compound.lower()
        fresh = min(laps, V4_THRESHOLDS[compound])
        worn = max(laps - V4_THRESHOLDS[compound], 0)
        cliff = max(laps - V4_CLIFF_THRESHOLDS[compound], 0)
        total += V4_PARAMS[f"{prefix}_offset"] * laps
        total += V4_PARAMS[f"{prefix}_fresh_linear"] * s1(fresh)
        total += V4_PARAMS[f"{prefix}_wear_linear"] * s1(worn)
        total += V4_PARAMS[f"{prefix}_wear_quadratic"] * s2(worn)
        total += V4_PARAMS[f"{prefix}_temp_wear"] * temp * s1(worn)
        total += V4_PARAMS[f"{prefix}_cliff_penalty"] * s1(cliff)
    return total


def feature_row(strategy, rc):
    stints = build_stints(strategy, rc["total_laps"])
    row = {
        "track": rc["track"],
        "starting_tire": strategy["starting_tire"],
        "tire_sequence": seq(strategy),
        "stint_1_compound": stints[0][0] if len(stints) > 0 else None,
        "stint_2_compound": stints[1][0] if len(stints) > 1 else None,
        "stint_3_compound": stints[2][0] if len(stints) > 2 else None,
        "total_laps": rc["total_laps"],
        "base_lap_time": rc["base_lap_time"],
        "pit_lane_time": rc["pit_lane_time"],
        "track_temp": rc["track_temp"],
        "pit_stop_count": len(strategy["pit_stops"]),
        "first_stop_lap": strategy["pit_stops"][0]["lap"] if strategy["pit_stops"] else None,
        "second_stop_lap": strategy["pit_stops"][1]["lap"] if len(strategy["pit_stops"]) > 1 else None,
        "soft_laps": 0,
        "medium_laps": 0,
        "hard_laps": 0,
        "soft_max_stint": 0,
        "medium_max_stint": 0,
        "hard_max_stint": 0,
        "stint_1_laps": stints[0][1] if len(stints) > 0 else 0,
        "stint_2_laps": stints[1][1] if len(stints) > 1 else 0,
        "stint_3_laps": stints[2][1] if len(stints) > 2 else 0,
    }
    for compound, laps in stints:
        prefix = compound.lower()
        row[f"{prefix}_laps"] += laps
        row[f"{prefix}_max_stint"] = max(row[f"{prefix}_max_stint"], laps)
    row["pred_v2"] = v2_score(stints, rc)
    row["pred_v4"] = v4_score(stints, rc)
    row["pred_blend"] = BLEND_ALPHA_V2 * row["pred_v2"] + (1.0 - BLEND_ALPHA_V2) * row["pred_v4"]
    return row


def encode(frame):
    for col in METADATA["categorical_columns"]:
        frame[col] = frame[col].fillna("__MISSING__").astype(str)
    numeric_cols = [c for c in METADATA["feature_columns"] if c not in METADATA["categorical_columns"]]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame = pd.get_dummies(frame, columns=METADATA["categorical_columns"], dummy_na=True)
    for col in METADATA["encoded_columns"]:
        if col not in frame.columns:
            frame[col] = 0
    return frame[METADATA["encoded_columns"]]


def main():
    failed = []
    for path in sorted(INPUT_DIR.glob("test_*.json")):
        try:
            test_case = json.loads(path.read_text())
            rows = []
            for grid_slot in sorted(test_case["strategies"]):
                strategy = test_case["strategies"][grid_slot]
                row = feature_row(strategy, test_case["race_config"])
                rows.append(row)
            frame = pd.DataFrame(rows)
            encoded = encode(frame)
            _ = RANKER.predict(encoded)
            print(path.name, "OK")
        except Exception as e:
            failed.append((path.name, repr(e)))
            print(path.name, "ERR", repr(e))
    print("failures", len(failed))


if __name__ == "__main__":
    main()
