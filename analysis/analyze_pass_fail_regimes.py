#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "test_cases" / "inputs"
EXPECTED_DIR = ROOT / "data" / "test_cases" / "expected_outputs"
SOLUTION = ROOT / "solution" / "race_simulator_xgb.py"
OUTPUT_DIR = ROOT / "analysis" / "pass_fail_regimes"
RACE_REPORT_PATH = OUTPUT_DIR / "race_report.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


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


def sequence_counts(race: dict) -> dict:
    counts = {}
    one_stop = 0
    two_stop = 0
    for strategy in race["strategies"].values():
        seq = ">".join([strategy["starting_tire"]] + [stop["to_tire"] for stop in strategy["pit_stops"]])
        counts[seq] = counts.get(seq, 0) + 1
        if len(strategy["pit_stops"]) == 1:
            one_stop += 1
        elif len(strategy["pit_stops"]) == 2:
            two_stop += 1
    top_sequences = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return {
        "unique_sequence_count": len(counts),
        "one_stop_drivers": one_stop,
        "two_stop_drivers": two_stop,
        "top_sequence_1": top_sequences[0][0] if top_sequences else "",
        "top_sequence_1_count": top_sequences[0][1] if top_sequences else 0,
        "top_sequence_2": top_sequences[1][0] if len(top_sequences) > 1 else "",
        "top_sequence_2_count": top_sequences[1][1] if len(top_sequences) > 1 else 0,
        "top_sequence_3": top_sequences[2][0] if len(top_sequences) > 2 else "",
        "top_sequence_3_count": top_sequences[2][1] if len(top_sequences) > 2 else 0,
    }


def analyze_case(input_path: Path) -> dict:
    race = json.loads(input_path.read_text())
    expected = json.loads((EXPECTED_DIR / input_path.name).read_text())
    pred = run_solver(input_path)

    true_order = expected["finishing_positions"]
    pred_order = pred["finishing_positions"]
    rc = race["race_config"]
    seq_stats = sequence_counts(race)

    return {
        "race_id": race["race_id"],
        "track": rc["track"],
        "total_laps": rc["total_laps"],
        "track_temp": rc["track_temp"],
        "pit_lane_time": rc["pit_lane_time"],
        "temp_band": temp_band(rc["track_temp"]),
        "lap_band": lap_band(rc["total_laps"]),
        "exact_match": int(true_order == pred_order),
        "first_match": int(true_order[0] == pred_order[0]),
        "top3_set_match": int(set(true_order[:3]) == set(pred_order[:3])),
        "top5_set_match": int(set(true_order[:5]) == set(pred_order[:5])),
        **seq_stats,
    }


def grouped_stats(df: pd.DataFrame, col: str) -> list[dict]:
    return (
        df.groupby(col)
        .agg(
            races=("race_id", "size"),
            exact_match_rate=("exact_match", "mean"),
            first_match_rate=("first_match", "mean"),
            top3_set_match_rate=("top3_set_match", "mean"),
            top5_set_match_rate=("top5_set_match", "mean"),
        )
        .round(4)
        .reset_index()
        .to_dict(orient="records")
    )


def main():
    race_rows = [analyze_case(path) for path in sorted(INPUT_DIR.glob("test_*.json"))]
    race_df = pd.DataFrame(race_rows).sort_values(["exact_match", "track", "race_id"], ascending=[False, True, True])
    passed = race_df[race_df["exact_match"] == 1].copy()
    failed = race_df[race_df["exact_match"] == 0].copy()

    summary = {
        "overall": {
            "races": int(len(race_df)),
            "passed": int(len(passed)),
            "failed": int(len(failed)),
            "exact_match_rate": float(race_df["exact_match"].mean()),
            "first_match_rate": float(race_df["first_match"].mean()),
            "top3_set_match_rate": float(race_df["top3_set_match"].mean()),
            "top5_set_match_rate": float(race_df["top5_set_match"].mean()),
        },
        "pass_vs_fail_means": {
            "passed_mean_total_laps": float(passed["total_laps"].mean()) if len(passed) else None,
            "failed_mean_total_laps": float(failed["total_laps"].mean()) if len(failed) else None,
            "passed_mean_track_temp": float(passed["track_temp"].mean()) if len(passed) else None,
            "failed_mean_track_temp": float(failed["track_temp"].mean()) if len(failed) else None,
            "passed_mean_pit_lane_time": float(passed["pit_lane_time"].mean()) if len(passed) else None,
            "failed_mean_pit_lane_time": float(failed["pit_lane_time"].mean()) if len(failed) else None,
            "passed_mean_unique_sequence_count": float(passed["unique_sequence_count"].mean()) if len(passed) else None,
            "failed_mean_unique_sequence_count": float(failed["unique_sequence_count"].mean()) if len(failed) else None,
        },
        "by_track": grouped_stats(race_df, "track"),
        "by_temp_band": grouped_stats(race_df, "temp_band"),
        "by_lap_band": grouped_stats(race_df, "lap_band"),
        "passed_cases": passed.to_dict(orient="records"),
        "failed_top_sequence_1": (
            failed.groupby("top_sequence_1")
            .agg(races=("race_id", "size"), exact_match_rate=("exact_match", "mean"))
            .sort_values("races", ascending=False)
            .round(4)
            .reset_index()
            .to_dict(orient="records")
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    race_df.to_csv(RACE_REPORT_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print("Overall:")
    print(json.dumps(summary["overall"], indent=2))
    print("\nPass vs fail means:")
    print(json.dumps(summary["pass_vs_fail_means"], indent=2))
    print("\nBy track:")
    print(pd.DataFrame(summary["by_track"]).to_string(index=False))
    print("\nBy temp band:")
    print(pd.DataFrame(summary["by_temp_band"]).to_string(index=False))
    print("\nBy lap band:")
    print(pd.DataFrame(summary["by_lap_band"]).to_string(index=False))
    print("\nPassed cases:")
    print(passed.to_string(index=False))
    print(f"\nWrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
