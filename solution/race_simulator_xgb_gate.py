#!/usr/bin/env python3
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from xgboost import XGBRanker


ROOT = Path(__file__).resolve().parent.parent
NO_COMP_DIR = ROOT / "analysis" / "models" / "xgb_ranker_no_comp"
COMP_DIR = ROOT / "analysis" / "models" / "xgb_ranker_comp"
BALANCED_DIR = ROOT / "analysis" / "models" / "xgb_ranker_balanced"
TIEBREAK_DIR = ROOT / "analysis" / "models" / "comp_tiebreaker"

BLEND_ALPHA_V2 = 0.6

V2_THRESHOLDS = {"SOFT": 4, "MEDIUM": 14, "HARD": 20}
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
V4_THRESHOLDS = {"SOFT": 5, "MEDIUM": 13, "HARD": 23}
V4_CLIFF_THRESHOLDS = {"SOFT": 7, "MEDIUM": 15, "HARD": 25}
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


def load_bundle(model_dir: Path):
    metadata = json.loads((model_dir / "metadata.json").read_text())
    ranker = XGBRanker()
    ranker.load_model(str(model_dir / "model.json"))
    return metadata, ranker


NO_COMP_META, NO_COMP_MODEL = load_bundle(NO_COMP_DIR)
COMP_META, COMP_MODEL = load_bundle(COMP_DIR)
HAS_BALANCED = BALANCED_DIR.exists()
if HAS_BALANCED:
    BALANCED_META, BALANCED_MODEL = load_bundle(BALANCED_DIR)
else:
    BALANCED_META, BALANCED_MODEL = None, None
HAS_TIEBREAK = TIEBREAK_DIR.exists()
if HAS_TIEBREAK:
    with (TIEBREAK_DIR / "model.pkl").open("rb") as fh:
        COMP_TIEBREAK_MODEL = pickle.load(fh)
    COMP_TIEBREAK_META = json.loads((TIEBREAK_DIR / "metadata.json").read_text())
else:
    COMP_TIEBREAK_MODEL = None
    COMP_TIEBREAK_META = {"pair_gap": 0.12, "adjustment_strength": 0.08}


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


def add_race_composition_features(frame, metadata):
    frame = frame.copy()
    major_sequences = metadata.get("major_sequences", [])
    race_size = float(len(frame))
    frame["race_size"] = race_size
    frame["unique_sequence_count"] = float(frame["tire_sequence"].nunique())
    seq_counts = frame["tire_sequence"].value_counts().to_dict()
    tire_counts = frame["starting_tire"].value_counts().to_dict()
    one_stop_count = float((frame["pit_stop_count"] == 1).sum())
    two_stop_count = float((frame["pit_stop_count"] == 2).sum())
    frame["sequence_field_count"] = frame["tire_sequence"].map(seq_counts).astype(float)
    frame["sequence_field_share"] = frame["sequence_field_count"] / race_size
    frame["starting_tire_field_count"] = frame["starting_tire"].map(tire_counts).astype(float)
    frame["starting_tire_field_share"] = frame["starting_tire_field_count"] / race_size
    frame["one_stop_field_count"] = one_stop_count
    frame["two_stop_field_count"] = two_stop_count
    frame["one_stop_field_share"] = one_stop_count / race_size
    frame["two_stop_field_share"] = two_stop_count / race_size
    for seq_name in major_sequences:
        safe = seq_name.lower().replace(">", "_to_")
        count_col = f"field_count__{safe}"
        share_col = f"field_share__{safe}"
        count = float(seq_counts.get(seq_name, 0.0))
        frame[count_col] = count
        frame[share_col] = count / race_size
    return frame


def encode(frame, metadata):
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


def apply_comp_tiebreak(frame):
    if COMP_TIEBREAK_MODEL is None:
        return frame
    pair_gap = float(COMP_TIEBREAK_META.get("pair_gap", 0.12))
    adjustment_strength = float(COMP_TIEBREAK_META.get("adjustment_strength", 0.08))
    race = frame.sort_values(["pred_score", "driver_id"], ascending=[True, True]).reset_index(drop=True).copy()
    adjustment = [0.0] * len(race)
    pair_rows = []
    pair_positions = []
    for i in range(len(race) - 1):
        j = i + 1
        if abs(float(race.loc[i, "pred_score"] - race.loc[j, "pred_score"])) > pair_gap:
            continue
        pair_rows.append(
            {
                "track": race.loc[i, "track"],
                "temp_band": race.loc[i, "temp_band"],
                "lap_band": race.loc[i, "lap_band"],
                "left_starting_tire": race.loc[i, "starting_tire"],
                "right_starting_tire": race.loc[j, "starting_tire"],
                "left_tire_sequence": race.loc[i, "tire_sequence"],
                "right_tire_sequence": race.loc[j, "tire_sequence"],
                "score_gap": float(race.loc[i, "pred_score"] - race.loc[j, "pred_score"]),
                "abs_score_gap": abs(float(race.loc[i, "pred_score"] - race.loc[j, "pred_score"])),
                "blend_gap": float(race.loc[i, "pred_blend"] - race.loc[j, "pred_blend"]),
                "v2_gap": float(race.loc[i, "pred_v2"] - race.loc[j, "pred_v2"]),
                "v4_gap": float(race.loc[i, "pred_v4"] - race.loc[j, "pred_v4"]),
                "pit_stop_gap": float(race.loc[i, "pit_stop_count"] - race.loc[j, "pit_stop_count"]),
                "first_stop_frac_gap": float(race.loc[i, "first_stop_frac"] - race.loc[j, "first_stop_frac"]),
                "second_stop_frac_gap": float(race.loc[i, "second_stop_frac"] - race.loc[j, "second_stop_frac"]),
                "soft_frac_gap": float(race.loc[i, "soft_frac"] - race.loc[j, "soft_frac"]),
                "medium_frac_gap": float(race.loc[i, "medium_frac"] - race.loc[j, "medium_frac"]),
                "hard_frac_gap": float(race.loc[i, "hard_frac"] - race.loc[j, "hard_frac"]),
                "stint_1_frac_gap": float(race.loc[i, "stint_1_frac"] - race.loc[j, "stint_1_frac"]),
                "stint_2_frac_gap": float(race.loc[i, "stint_2_frac"] - race.loc[j, "stint_2_frac"]),
                "stint_3_frac_gap": float(race.loc[i, "stint_3_frac"] - race.loc[j, "stint_3_frac"]),
                "sequence_field_share_gap": float(race.loc[i, "sequence_field_share"] - race.loc[j, "sequence_field_share"]),
                "starting_tire_field_share_gap": float(race.loc[i, "starting_tire_field_share"] - race.loc[j, "starting_tire_field_share"]),
                "track_temp": float(race.loc[i, "track_temp"]),
                "total_laps": float(race.loc[i, "total_laps"]),
                "pit_lane_time": float(race.loc[i, "pit_lane_time"]),
            }
        )
        pair_positions.append((i, j))
    if pair_rows:
        probs = COMP_TIEBREAK_MODEL.predict_proba(pd.DataFrame(pair_rows))[:, 1]
        for (i, j), prob_left_ahead in zip(pair_positions, probs):
            centered = float(prob_left_ahead) - 0.5
            adjustment[i] -= adjustment_strength * centered
            adjustment[j] += adjustment_strength * centered
    race["pred_score"] = race["pred_score"] + adjustment
    return race


def choose_bundle(frame):
    unique_sequences = int(frame["tire_sequence"].nunique())
    two_stop = int((frame["pit_stop_count"] == 2).sum())
    total_laps = float(frame["total_laps"].iloc[0])
    lap_band = str(frame["lap_band"].iloc[0])
    temp_band = str(frame["temp_band"].iloc[0])
    top_count = int(frame["tire_sequence"].value_counts().iloc[0])
    top_sequence = str(frame["tire_sequence"].value_counts().index[0])

    dominant_soft_hard = top_count >= 6 and top_sequence == "SOFT>HARD"
    dominant_hard_medium_short = top_count >= 6 and top_sequence == "HARD>MEDIUM" and total_laps <= 35
    dominant_medium_hard_short = top_count >= 6 and top_sequence == "MEDIUM>HARD" and total_laps <= 35

    balanced_cluster = (
        two_stop == 0
        and 4 <= unique_sequences <= 6
        and 28 <= float(frame["track_temp"].iloc[0]) <= 36
        and total_laps <= 45
        and 4 <= top_count <= 10
    )

    if HAS_BALANCED and balanced_cluster and top_sequence in {"MEDIUM>HARD", "SOFT>HARD", "SOFT>MEDIUM", "MEDIUM>SOFT"}:
        return "balanced", BALANCED_META, BALANCED_MODEL
    if total_laps <= 35 and unique_sequences <= 4:
        return "no_comp", NO_COMP_META, NO_COMP_MODEL
    if dominant_soft_hard and two_stop == 0 and unique_sequences <= 6 and total_laps <= 55:
        return "no_comp", NO_COMP_META, NO_COMP_MODEL
    if dominant_hard_medium_short and two_stop == 0 and unique_sequences <= 5 and temp_band in {"mild", "warm"}:
        return "no_comp", NO_COMP_META, NO_COMP_MODEL
    if dominant_medium_hard_short and two_stop == 0 and unique_sequences <= 5:
        return "no_comp", NO_COMP_META, NO_COMP_MODEL
    if two_stop >= 5:
        return "comp", COMP_META, COMP_MODEL
    if top_sequence == "HARD>MEDIUM" and lap_band in {"mid", "long"}:
        return "comp", COMP_META, COMP_MODEL
    if total_laps >= 48:
        return "comp", COMP_META, COMP_MODEL
    if unique_sequences >= 7:
        return "comp", COMP_META, COMP_MODEL
    return "comp", COMP_META, COMP_MODEL


def predict_finishing_positions(test_case):
    rows = []
    for grid_slot in sorted(test_case["strategies"]):
        strategy = test_case["strategies"][grid_slot]
        row = feature_row(strategy, test_case["race_config"])
        row["driver_id"] = strategy["driver_id"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    branch, metadata, model = choose_bundle(frame)
    if metadata.get("use_composition_features", False):
        frame = add_race_composition_features(frame, metadata)
    encoded = encode(frame, metadata)
    frame["pred_score"] = model.predict(encoded)
    if branch == "comp":
        frame = apply_comp_tiebreak(frame)
    frame = frame.sort_values(["pred_score", "driver_id"], ascending=[True, True])
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
