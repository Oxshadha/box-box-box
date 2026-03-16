#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRanker


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "analysis" / "models" / "xgb_ranker"
MODEL_PATH = MODEL_DIR / "model.json"
METADATA_PATH = MODEL_DIR / "metadata.json"
RULE_DIR = ROOT / "analysis" / "rule_mining"
SEQUENCE_RULES_PATH = RULE_DIR / "sequence_dominance_rules.csv"
STOP_RULES_PATH = RULE_DIR / "one_stop_timing_rules.csv"

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

SEQUENCE_RULES = pd.read_csv(SEQUENCE_RULES_PATH)
STOP_RULES = pd.read_csv(STOP_RULES_PATH)

SEQUENCE_RULE_MAP = {
    (row.temp_band, row.lap_band, row.winner_sequence, row.loser_sequence): {
        "winner_win_rate": float(row.winner_win_rate),
        "pairings": int(row.pairings),
    }
    for row in SEQUENCE_RULES.itertuples(index=False)
}
STOP_RULE_MAP = {
    (row.tire_sequence, row.temp_band, row.lap_band, row.first_stop_frac_bin): {
        "quality_score": float(row.quality_score),
        "rows": int(row.rows),
    }
    for row in STOP_RULES.itertuples(index=False)
}


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


def v2_score(stints, rc):
    total = rc["base_lap_time"] * rc["total_laps"] + rc["pit_lane_time"] * (len(stints) - 1)
    temp = rc["track_temp"]
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


def v4_score(stints, rc):
    total = rc["base_lap_time"] * rc["total_laps"] + rc["pit_lane_time"] * (len(stints) - 1)
    temp = rc["track_temp"]
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
    total_laps = float(rc["total_laps"])
    track_temp = float(rc["track_temp"])
    first_stop_lap = float(strategy["pit_stops"][0]["lap"] if strategy["pit_stops"] else 0.0)
    second_stop_lap = float(strategy["pit_stops"][1]["lap"] if len(strategy["pit_stops"]) > 1 else 0.0)
    if track_temp <= 24:
        temp_band = "cool"
    elif track_temp <= 30:
        temp_band = "mild"
    elif track_temp <= 36:
        temp_band = "warm"
    else:
        temp_band = "hot"
    if total_laps <= 35:
        lap_band = "short"
    elif total_laps <= 45:
        lap_band = "mid_short"
    elif total_laps <= 55:
        lap_band = "mid"
    else:
        lap_band = "long"
    tire_sequence = seq(strategy)
    row = {
        "track": rc["track"],
        "starting_tire": strategy["starting_tire"],
        "tire_sequence": tire_sequence,
        "stint_1_compound": stints[0][0] if len(stints) > 0 else None,
        "stint_2_compound": stints[1][0] if len(stints) > 1 else None,
        "stint_3_compound": stints[2][0] if len(stints) > 2 else None,
        "temp_band": temp_band,
        "lap_band": lap_band,
        "sequence_temp_band": f"{tire_sequence}|{temp_band}",
        "sequence_lap_band": f"{tire_sequence}|{lap_band}",
        "start_tire_lap_band": f"{strategy['starting_tire']}|{lap_band}",
        "start_tire_temp_band": f"{strategy['starting_tire']}|{temp_band}",
        "total_laps": total_laps,
        "base_lap_time": rc["base_lap_time"],
        "pit_lane_time": rc["pit_lane_time"],
        "track_temp": track_temp,
        "pit_stop_count": len(strategy["pit_stops"]),
        "first_stop_lap": first_stop_lap,
        "second_stop_lap": second_stop_lap,
        "first_stop_frac": first_stop_lap / total_laps,
        "second_stop_frac": second_stop_lap / total_laps,
        "soft_laps": 0.0,
        "medium_laps": 0.0,
        "hard_laps": 0.0,
        "soft_frac": 0.0,
        "medium_frac": 0.0,
        "hard_frac": 0.0,
        "soft_max_stint": 0.0,
        "medium_max_stint": 0.0,
        "hard_max_stint": 0.0,
        "stint_1_laps": float(stints[0][1] if len(stints) > 0 else 0),
        "stint_2_laps": float(stints[1][1] if len(stints) > 1 else 0),
        "stint_3_laps": float(stints[2][1] if len(stints) > 2 else 0),
        "stint_1_frac": float(stints[0][1] if len(stints) > 0 else 0) / total_laps,
        "stint_2_frac": float(stints[1][1] if len(stints) > 1 else 0) / total_laps,
        "stint_3_frac": float(stints[2][1] if len(stints) > 2 else 0) / total_laps,
    }
    for compound, laps in stints:
        prefix = compound.lower()
        row[f"{prefix}_laps"] += float(laps)
        row[f"{prefix}_max_stint"] = max(row[f"{prefix}_max_stint"], float(laps))
    row["soft_frac"] = row["soft_laps"] / total_laps
    row["medium_frac"] = row["medium_laps"] / total_laps
    row["hard_frac"] = row["hard_laps"] / total_laps
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


def stop_frac_bin(value):
    if pd.isna(value) or value <= 0:
        return None
    if value <= 0.2:
        return "0.0-0.2"
    if value <= 0.3:
        return "0.2-0.3"
    if value <= 0.4:
        return "0.3-0.4"
    if value <= 0.5:
        return "0.4-0.5"
    if value <= 0.7:
        return "0.5-0.7"
    return "0.7-1.0"


def apply_rule_rerank(frame):
    frame = frame.copy()
    spread = float(frame["pred_score"].max() - frame["pred_score"].min())
    local_scale = max(spread / 80.0, 0.02)
    bonuses = np.zeros(len(frame), dtype=float)

    # Reward strategies whose stop timing matches historically strong bins.
    for idx, row in enumerate(frame.itertuples(index=False)):
        frac_bin = stop_frac_bin(row.first_stop_frac)
        if frac_bin is None:
            continue
        key = (row.tire_sequence, row.temp_band, row.lap_band, frac_bin)
        rule = STOP_RULE_MAP.get(key)
        if not rule:
            continue
        quality = max(0.0, min(1.0, rule["quality_score"]))
        support = min(rule["rows"], 1500) / 1500.0
        bonuses[idx] += 0.45 * local_scale * quality * support

    # Apply pairwise regime rules, strongest when model scores are close.
    rows = list(frame.itertuples(index=False))
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            left = rows[i]
            right = rows[j]
            gap = abs(left.pred_score - right.pred_score)
            closeness = max(0.0, 1.0 - gap / (4.0 * local_scale))
            if closeness <= 0:
                continue

            forward = SEQUENCE_RULE_MAP.get(
                (left.temp_band, left.lap_band, left.tire_sequence, right.tire_sequence)
            )
            backward = SEQUENCE_RULE_MAP.get(
                (left.temp_band, left.lap_band, right.tire_sequence, left.tire_sequence)
            )

            if forward:
                strength = max(0.0, min(0.49, forward["winner_win_rate"] - 0.5))
                support = min(forward["pairings"], 8000) / 8000.0
                delta = 0.90 * local_scale * closeness * strength * support
                bonuses[i] += delta
                bonuses[j] -= delta
            elif backward:
                strength = max(0.0, min(0.49, backward["winner_win_rate"] - 0.5))
                support = min(backward["pairings"], 8000) / 8000.0
                delta = 0.90 * local_scale * closeness * strength * support
                bonuses[i] -= delta
                bonuses[j] += delta

    frame["adjusted_score"] = frame["pred_score"] - bonuses
    return frame.sort_values(["adjusted_score", "pred_score", "driver_id"], ascending=[True, True, True])


def predict_finishing_positions(test_case):
    rows = []
    for grid_slot in sorted(test_case["strategies"]):
        strategy = test_case["strategies"][grid_slot]
        row = feature_row(strategy, test_case["race_config"])
        row["driver_id"] = strategy["driver_id"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    encoded = encode(frame.copy())
    frame["pred_score"] = RANKER.predict(encoded)
    frame = apply_rule_rerank(frame)
    return frame["driver_id"].tolist()


def main():
    test_case = json.load(sys.stdin)
    output = {
        "race_id": test_case["race_id"],
        "finishing_positions": predict_finishing_positions(test_case),
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
