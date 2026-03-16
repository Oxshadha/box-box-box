#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "test_cases" / "inputs"
EXPECTED_DIR = ROOT / "data" / "test_cases" / "expected_outputs"
SOLUTION = ROOT / "solution" / "race_simulator.py"
OUTPUT_DIR = ROOT / "analysis" / "test_failure_analysis"
RACE_REPORT = OUTPUT_DIR / "race_report.csv"
DRIVER_REPORT = OUTPUT_DIR / "driver_report.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


def run_solver(input_path: Path) -> dict:
    with input_path.open("rb") as fh:
        try:
            completed = subprocess.run(
                [sys.executable, str(SOLUTION)],
                stdin=fh,
                capture_output=True,
                check=True,
                cwd=ROOT,
            )
        except subprocess.CalledProcessError as exc:
            stderr_text = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Solver failed for {input_path.name} with interpreter {sys.executable}. "
                f"Stderr: {stderr_text}"
            ) from exc
    return json.loads(completed.stdout)


def analyze_case(input_path: Path):
    expected_path = EXPECTED_DIR / input_path.name
    prediction = run_solver(input_path)
    expected = json.loads(expected_path.read_text())
    race = json.loads(input_path.read_text())

    pred_order = prediction["finishing_positions"]
    true_order = expected["finishing_positions"]
    true_rank = {driver: idx + 1 for idx, driver in enumerate(true_order)}
    pred_rank = {driver: idx + 1 for idx, driver in enumerate(pred_order)}

    driver_rows = []
    for grid_slot, strategy in sorted(race["strategies"].items()):
        driver_id = strategy["driver_id"]
        driver_rows.append(
            {
                "race_id": race["race_id"],
                "track": race["race_config"]["track"],
                "driver_id": driver_id,
                "starting_tire": strategy["starting_tire"],
                "pit_stop_count": len(strategy["pit_stops"]),
                "pit_laps": ">".join(str(stop["lap"]) for stop in strategy["pit_stops"]),
                "tire_sequence": ">".join(
                    [strategy["starting_tire"]] + [stop["to_tire"] for stop in strategy["pit_stops"]]
                ),
                "true_rank": true_rank[driver_id],
                "pred_rank": pred_rank[driver_id],
                "rank_error": pred_rank[driver_id] - true_rank[driver_id],
                "abs_rank_error": abs(pred_rank[driver_id] - true_rank[driver_id]),
            }
        )

    top3_match = set(pred_order[:3]) == set(true_order[:3])
    top5_match = set(pred_order[:5]) == set(true_order[:5])
    race_row = {
        "race_id": race["race_id"],
        "track": race["race_config"]["track"],
        "total_laps": race["race_config"]["total_laps"],
        "track_temp": race["race_config"]["track_temp"],
        "pit_lane_time": race["race_config"]["pit_lane_time"],
        "exact_match": int(pred_order == true_order),
        "top3_match": int(top3_match),
        "top5_match": int(top5_match),
        "kendall_tau": float(kendalltau(list(true_rank.values()), [pred_rank[d] for d in true_order]).statistic),
        "spearman_rho": float(spearmanr(list(true_rank.values()), [pred_rank[d] for d in true_order]).statistic),
        "mean_abs_rank_error": float(np.mean([row["abs_rank_error"] for row in driver_rows])),
        "max_abs_rank_error": int(max(row["abs_rank_error"] for row in driver_rows)),
        "first_mismatch_at": next((i + 1 for i, (a, b) in enumerate(zip(true_order, pred_order)) if a != b), 21),
    }
    return race_row, driver_rows


def main():
    race_rows = []
    driver_rows = []
    for input_path in sorted(INPUT_DIR.glob("test_*.json")):
        race_row, case_driver_rows = analyze_case(input_path)
        race_rows.append(race_row)
        driver_rows.extend(case_driver_rows)

    race_df = pd.DataFrame(race_rows).sort_values(["exact_match", "kendall_tau", "mean_abs_rank_error"], ascending=[True, True, False])
    driver_df = pd.DataFrame(driver_rows)

    summary = {
        "overall": {
            "races": int(len(race_df)),
            "exact_match_rate": float(race_df["exact_match"].mean()),
            "top3_match_rate": float(race_df["top3_match"].mean()),
            "top5_match_rate": float(race_df["top5_match"].mean()),
            "mean_kendall_tau": float(race_df["kendall_tau"].mean()),
            "mean_spearman_rho": float(race_df["spearman_rho"].mean()),
            "mean_abs_rank_error": float(race_df["mean_abs_rank_error"].mean()),
        },
        "by_track": (
            race_df.groupby("track")
            .agg(
                races=("race_id", "size"),
                exact_match_rate=("exact_match", "mean"),
                top3_match_rate=("top3_match", "mean"),
                top5_match_rate=("top5_match", "mean"),
                mean_kendall_tau=("kendall_tau", "mean"),
                mean_abs_rank_error=("mean_abs_rank_error", "mean"),
            )
            .round(4)
            .reset_index()
            .to_dict(orient="records")
        ),
        "worst_sequences": (
            driver_df.groupby("tire_sequence")
            .agg(
                rows=("driver_id", "size"),
                mean_abs_rank_error=("abs_rank_error", "mean"),
                median_abs_rank_error=("abs_rank_error", "median"),
                mean_rank_error=("rank_error", "mean"),
            )
            .sort_values("mean_abs_rank_error", ascending=False)
            .head(12)
            .round(4)
            .reset_index()
            .to_dict(orient="records")
        ),
        "worst_starters": (
            driver_df.groupby("starting_tire")
            .agg(
                rows=("driver_id", "size"),
                mean_abs_rank_error=("abs_rank_error", "mean"),
                mean_rank_error=("rank_error", "mean"),
            )
            .sort_values("mean_abs_rank_error", ascending=False)
            .round(4)
            .reset_index()
            .to_dict(orient="records")
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    race_df.to_csv(RACE_REPORT, index=False)
    driver_df.to_csv(DRIVER_REPORT, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print("Overall summary:")
    print(json.dumps(summary["overall"], indent=2))
    print("\nBy track:")
    print(pd.DataFrame(summary["by_track"]).to_string(index=False))
    print("\nWorst sequences:")
    print(pd.DataFrame(summary["worst_sequences"]).to_string(index=False))
    print("\nWorst starters:")
    print(pd.DataFrame(summary["worst_starters"]).to_string(index=False))
    print("\nWorst races:")
    print(race_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
