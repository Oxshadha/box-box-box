#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "test_cases" / "inputs"
EXPECTED_DIR = ROOT / "data" / "test_cases" / "expected_outputs"
SOLUTION = ROOT / "solution" / "race_simulator_xgb.py"
OUTPUT_DIR = ROOT / "analysis" / "forensic_case_compare"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
PAIR_REPORT_PATH = OUTPUT_DIR / "nearest_pass_fail_pairs.csv"


def temp_band(track_temp: float) -> str:
    if track_temp <= 24:
        return "cool"
    if track_temp <= 30:
        return "mild"
    if track_temp <= 36:
        return "warm"
    return "hot"


def lap_band(total_laps: int) -> str:
    if total_laps <= 35:
        return "short"
    if total_laps <= 45:
        return "mid_short"
    if total_laps <= 55:
        return "mid"
    return "long"


def run_solver(input_path: Path) -> dict:
    with input_path.open("rb") as fh:
        completed = subprocess.run(
            [sys.executable, str(SOLUTION)],
            stdin=fh,
            capture_output=True,
            check=True,
            cwd=ROOT,
        )
    return json.loads(completed.stdout)


def strategy_counts(race: dict) -> dict:
    counts = {}
    two_stop = 0
    for strategy in race["strategies"].values():
        sequence = ">".join([strategy["starting_tire"]] + [stop["to_tire"] for stop in strategy["pit_stops"]])
        counts[sequence] = counts.get(sequence, 0) + 1
        if len(strategy["pit_stops"]) == 2:
            two_stop += 1
    return counts, two_stop


def top_sequence_signature(counts: dict, top_n: int = 5) -> dict:
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top_n]
    return {k: v for k, v in items}


def summarize_case(input_path: Path) -> dict:
    race = json.loads(input_path.read_text())
    expected = json.loads((EXPECTED_DIR / input_path.name).read_text())
    pred = run_solver(input_path)
    rc = race["race_config"]
    counts, two_stop = strategy_counts(race)
    expected_order = expected["finishing_positions"]
    pred_order = pred["finishing_positions"]
    first_mismatch_at = next((i + 1 for i, (a, b) in enumerate(zip(expected_order, pred_order)) if a != b), 21)
    return {
        "race_id": race["race_id"],
        "track": rc["track"],
        "total_laps": rc["total_laps"],
        "track_temp": rc["track_temp"],
        "pit_lane_time": rc["pit_lane_time"],
        "temp_band": temp_band(rc["track_temp"]),
        "lap_band": lap_band(rc["total_laps"]),
        "exact_match": int(expected_order == pred_order),
        "first_match": int(expected_order[0] == pred_order[0]),
        "top3_set_match": int(set(expected_order[:3]) == set(pred_order[:3])),
        "top5_set_match": int(set(expected_order[:5]) == set(pred_order[:5])),
        "first_mismatch_at": first_mismatch_at,
        "unique_sequence_count": len(counts),
        "two_stop_drivers": two_stop,
        "top_sequence_signature": top_sequence_signature(counts),
        "expected_top5": expected_order[:5],
        "pred_top5": pred_order[:5],
    }


def sequence_distance(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return float(sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys))


def pair_rows(df: pd.DataFrame) -> list[dict]:
    passed = df[df["exact_match"] == 1].copy()
    failed = df[df["exact_match"] == 0].copy()
    rows = []
    for pass_row in passed.itertuples(index=False):
        candidate = failed.copy()
        candidate["track_match"] = (candidate["track"] == pass_row.track).astype(int)
        candidate["lap_band_match"] = (candidate["lap_band"] == pass_row.lap_band).astype(int)
        candidate["temp_band_match"] = (candidate["temp_band"] == pass_row.temp_band).astype(int)
        candidate["lap_gap"] = (candidate["total_laps"] - pass_row.total_laps).abs()
        candidate["temp_gap"] = (candidate["track_temp"] - pass_row.track_temp).abs()
        candidate["pit_gap"] = (candidate["pit_lane_time"] - pass_row.pit_lane_time).abs()
        candidate["seq_gap"] = candidate["top_sequence_signature"].apply(
            lambda sig: sequence_distance(sig, pass_row.top_sequence_signature)
        )
        candidate["distance"] = (
            8.0 * (1 - candidate["track_match"])
            + 3.0 * (1 - candidate["lap_band_match"])
            + 2.0 * (1 - candidate["temp_band_match"])
            + candidate["lap_gap"]
            + 0.75 * candidate["temp_gap"]
            + 5.0 * candidate["pit_gap"]
            + 0.8 * candidate["seq_gap"]
        )
        nearest = candidate.sort_values(["distance", "first_mismatch_at", "unique_sequence_count"]).head(3)
        for fail_row in nearest.itertuples(index=False):
            rows.append(
                {
                    "passed_race_id": pass_row.race_id,
                    "failed_race_id": fail_row.race_id,
                    "track_pair": f"{pass_row.track} | {fail_row.track}",
                    "lap_pair": f"{pass_row.total_laps} | {fail_row.total_laps}",
                    "temp_pair": f"{pass_row.track_temp} | {fail_row.track_temp}",
                    "pit_pair": f"{pass_row.pit_lane_time} | {fail_row.pit_lane_time}",
                    "band_pair": f"{pass_row.lap_band}/{pass_row.temp_band} | {fail_row.lap_band}/{fail_row.temp_band}",
                    "unique_sequence_pair": f"{pass_row.unique_sequence_count} | {fail_row.unique_sequence_count}",
                    "two_stop_pair": f"{pass_row.two_stop_drivers} | {fail_row.two_stop_drivers}",
                    "first_mismatch_at_failed": fail_row.first_mismatch_at,
                    "seq_gap": fail_row.seq_gap,
                    "distance": round(float(fail_row.distance), 4),
                    "passed_signature": json.dumps(pass_row.top_sequence_signature, sort_keys=True),
                    "failed_signature": json.dumps(fail_row.top_sequence_signature, sort_keys=True),
                    "passed_expected_top5": ",".join(pass_row.expected_top5),
                    "failed_expected_top5": ",".join(fail_row.expected_top5),
                    "failed_pred_top5": ",".join(fail_row.pred_top5),
                }
            )
    return rows


def main():
    cases = [summarize_case(path) for path in sorted(INPUT_DIR.glob("test_*.json"))]
    df = pd.DataFrame(cases)
    pair_df = pd.DataFrame(pair_rows(df))
    summary = {
        "overall": {
            "cases": int(len(df)),
            "passed": int(df["exact_match"].sum()),
            "failed": int((1 - df["exact_match"]).sum()),
        },
        "passed_cases": df[df["exact_match"] == 1].to_dict(orient="records"),
        "nearest_pairs": pair_df.to_dict(orient="records"),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(PAIR_REPORT_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print("Passed cases:")
    print(df[df["exact_match"] == 1][["race_id", "track", "total_laps", "track_temp", "lap_band", "temp_band", "unique_sequence_count", "two_stop_drivers"]].to_string(index=False))
    print("\nNearest pass/fail pairs:")
    print(pair_df.head(20).to_string(index=False))
    print(f"\nWrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
