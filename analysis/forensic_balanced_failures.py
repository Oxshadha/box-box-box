#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "test_cases" / "inputs"
EXPECTED_DIR = ROOT / "data" / "test_cases" / "expected_outputs"
SNAPSHOT_PATH = ROOT / "analysis" / "solver_snapshots" / "gate27.json"
OUTPUT_DIR = ROOT / "analysis" / "balanced_forensics"


def lap_band(total_laps: int) -> str:
    if total_laps <= 35:
        return "short"
    if total_laps <= 45:
        return "mid_short"
    if total_laps <= 55:
        return "mid"
    return "long"


def temp_band(track_temp: float) -> str:
    if track_temp <= 24:
        return "cool"
    if track_temp <= 30:
        return "mild"
    if track_temp <= 36:
        return "warm"
    return "hot"


def route_branch(race: dict) -> str:
    strategies = list(race["strategies"].values())
    sequences = [
        ">".join([s["starting_tire"]] + [stop["to_tire"] for stop in s["pit_stops"]])
        for s in strategies
    ]
    counts = Counter(sequences)
    unique_sequences = len(counts)
    top_sequence, top_count = counts.most_common(1)[0]
    two_stop = sum(1 for s in strategies if len(s["pit_stops"]) == 2)
    total_laps = int(race["race_config"]["total_laps"])
    track_temp = float(race["race_config"]["track_temp"])

    dominant_soft_hard = top_count >= 6 and top_sequence == "SOFT>HARD"
    dominant_hard_medium_short = top_count >= 6 and top_sequence == "HARD>MEDIUM" and total_laps <= 35
    dominant_medium_hard_short = top_count >= 6 and top_sequence == "MEDIUM>HARD" and total_laps <= 35
    balanced_cluster = (
        two_stop == 0
        and 4 <= unique_sequences <= 6
        and 28 <= track_temp <= 36
        and total_laps <= 45
        and 4 <= top_count <= 10
        and top_sequence in {"MEDIUM>HARD", "SOFT>HARD", "SOFT>MEDIUM", "MEDIUM>SOFT"}
    )
    if balanced_cluster:
        return "balanced"
    if total_laps <= 35 and unique_sequences <= 4:
        return "no_comp"
    if dominant_soft_hard and two_stop == 0 and unique_sequences <= 6 and total_laps <= 55:
        return "no_comp"
    if dominant_hard_medium_short and two_stop == 0 and unique_sequences <= 5 and temp_band(track_temp) in {"mild", "warm"}:
        return "no_comp"
    if dominant_medium_hard_short and two_stop == 0 and unique_sequences <= 5:
        return "no_comp"
    return "comp"


def summarize_case(race: dict, snapshot_row: dict) -> dict:
    strategies = list(race["strategies"].values())
    sequences = [
        ">".join([s["starting_tire"]] + [stop["to_tire"] for stop in s["pit_stops"]])
        for s in strategies
    ]
    counts = Counter(sequences)
    top_sequence, top_count = counts.most_common(1)[0]
    expected = json.loads((EXPECTED_DIR / f"{race['race_id'].lower()}.json").read_text())["finishing_positions"]
    pred = snapshot_row["pred_top5"].split(",")
    return {
        "race_id": race["race_id"],
        "track": race["race_config"]["track"],
        "total_laps": int(race["race_config"]["total_laps"]),
        "track_temp": float(race["race_config"]["track_temp"]),
        "pit_lane_time": float(race["race_config"]["pit_lane_time"]),
        "lap_band": lap_band(int(race["race_config"]["total_laps"])),
        "temp_band": temp_band(float(race["race_config"]["track_temp"])),
        "branch": route_branch(race),
        "exact_match": int(snapshot_row["exact_match"]),
        "front5_correct": int(snapshot_row["expected_top5"] == snapshot_row["pred_top5"]),
        "unique_sequence_count": len(counts),
        "top_sequence": top_sequence,
        "top_sequence_count": top_count,
        "two_stop_drivers": sum(1 for s in strategies if len(s["pit_stops"]) == 2),
        "sequence_signature": dict(counts.most_common(6)),
        "expected_top5": snapshot_row["expected_top5"],
        "pred_top5": snapshot_row["pred_top5"],
        "expected_full": expected,
    }


def signature_distance(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return float(sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys))


def main():
    snapshot_rows = {row["race_id"]: row for row in json.loads(SNAPSHOT_PATH.read_text())}
    cases = []
    for input_path in sorted(INPUT_DIR.glob("test_*.json")):
        race = json.loads(input_path.read_text())
        cases.append(summarize_case(race, snapshot_rows[race["race_id"]]))
    df = pd.DataFrame(cases)

    balanced = df[df["branch"] == "balanced"].copy()
    balanced_failed = balanced[balanced["exact_match"] == 0].copy()
    balanced_passed = balanced[balanced["exact_match"] == 1].copy()

    front5_failed = balanced_failed[balanced_failed["front5_correct"] == 1].copy()
    front5_failed = front5_failed.sort_values(["track", "lap_band", "temp_band", "top_sequence", "race_id"])

    nearest_rows = []
    for fail_row in front5_failed.itertuples(index=False):
        candidates = balanced_passed.copy()
        candidates["track_match"] = (candidates["track"] == fail_row.track).astype(int)
        candidates["lap_match"] = (candidates["lap_band"] == fail_row.lap_band).astype(int)
        candidates["temp_match"] = (candidates["temp_band"] == fail_row.temp_band).astype(int)
        candidates["top_match"] = (candidates["top_sequence"] == fail_row.top_sequence).astype(int)
        candidates["lap_gap"] = (candidates["total_laps"] - fail_row.total_laps).abs()
        candidates["temp_gap"] = (candidates["track_temp"] - fail_row.track_temp).abs()
        candidates["seq_gap"] = candidates["sequence_signature"].apply(lambda sig: signature_distance(sig, fail_row.sequence_signature))
        candidates["distance"] = (
            8.0 * (1 - candidates["track_match"])
            + 3.0 * (1 - candidates["lap_match"])
            + 2.0 * (1 - candidates["temp_match"])
            + 2.0 * (1 - candidates["top_match"])
            + candidates["lap_gap"]
            + 0.75 * candidates["temp_gap"]
            + 0.6 * candidates["seq_gap"]
        )
        nearest = candidates.sort_values(["distance", "lap_gap", "temp_gap"]).head(3)
        for pass_row in nearest.itertuples(index=False):
            nearest_rows.append(
                {
                    "failed_race_id": fail_row.race_id,
                    "passed_race_id": pass_row.race_id,
                    "failed_track": fail_row.track,
                    "passed_track": pass_row.track,
                    "failed_laps": fail_row.total_laps,
                    "passed_laps": pass_row.total_laps,
                    "failed_temp": fail_row.track_temp,
                    "passed_temp": pass_row.track_temp,
                    "failed_top_sequence": fail_row.top_sequence,
                    "passed_top_sequence": pass_row.top_sequence,
                    "failed_signature": json.dumps(fail_row.sequence_signature, sort_keys=True),
                    "passed_signature": json.dumps(pass_row.sequence_signature, sort_keys=True),
                    "distance": round(float(pass_row.distance), 4),
                }
            )

    summary = {
        "overall_balanced": {
            "cases": int(len(balanced)),
            "passed": int(balanced["exact_match"].sum()),
            "failed": int(len(balanced_failed)),
            "front5_correct_failures": int(len(front5_failed)),
        },
        "failed_by_track": (
            balanced_failed.groupby("track").size().sort_values(ascending=False).rename("races").reset_index().to_dict(orient="records")
        ),
        "failed_by_top_sequence": (
            balanced_failed.groupby("top_sequence").size().sort_values(ascending=False).rename("races").reset_index().to_dict(orient="records")
        ),
        "front5_correct_failures": front5_failed[
            [
                "race_id",
                "track",
                "total_laps",
                "track_temp",
                "lap_band",
                "temp_band",
                "top_sequence",
                "top_sequence_count",
                "unique_sequence_count",
                "expected_top5",
                "pred_top5",
            ]
        ].to_dict(orient="records"),
        "nearest_passed_pairs": nearest_rows,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(nearest_rows).to_csv(OUTPUT_DIR / "nearest_pairs.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
