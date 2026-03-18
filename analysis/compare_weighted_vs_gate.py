#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "test_cases" / "inputs"
EXPECTED_DIR = ROOT / "data" / "test_cases" / "expected_outputs"
SNAPSHOT_DIR = ROOT / "analysis" / "solver_snapshots"


def run_solver(solver_path: Path, input_path: Path) -> dict:
    with input_path.open("rb") as fh:
        completed = subprocess.run(
            [sys.executable, str(solver_path)],
            stdin=fh,
            capture_output=True,
            check=True,
            cwd=ROOT,
        )
    return json.loads(completed.stdout)


def make_snapshot(name: str, solver_path: Path) -> Path:
    rows = []
    for input_path in sorted(INPUT_DIR.glob("test_*.json")):
        race = json.loads(input_path.read_text())
        expected = json.loads((EXPECTED_DIR / input_path.name).read_text())
        pred = run_solver(solver_path, input_path)
        rows.append(
            {
                "race_id": race["race_id"],
                "track": race["race_config"]["track"],
                "total_laps": race["race_config"]["total_laps"],
                "track_temp": race["race_config"]["track_temp"],
                "pit_lane_time": race["race_config"]["pit_lane_time"],
                "expected_top5": ",".join(expected["finishing_positions"][:5]),
                "pred_top5": ",".join(pred["finishing_positions"][:5]),
                "exact_match": int(expected["finishing_positions"] == pred["finishing_positions"]),
            }
        )
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out = SNAPSHOT_DIR / f"{name}.json"
    out.write_text(json.dumps(rows, indent=2))
    return out


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: compare_weighted_vs_gate.py WEIGHTED_NAME SOLVER_PATH")
    weighted_name = sys.argv[1]
    solver_path = Path(sys.argv[2]).resolve()
    weighted_path = make_snapshot(weighted_name, solver_path)

    gate27 = pd.DataFrame(json.loads((SNAPSHOT_DIR / "gate27.json").read_text())).rename(
        columns={"exact_match": "exact_gate27", "pred_top5": "pred_top5_gate27"}
    )
    weighted = pd.DataFrame(json.loads(weighted_path.read_text())).rename(
        columns={"exact_match": "exact_weighted", "pred_top5": "pred_top5_weighted"}
    )
    merged = gate27.merge(
        weighted[["race_id", "exact_weighted", "pred_top5_weighted"]],
        on="race_id",
        how="inner",
    )
    gained = merged[(merged["exact_gate27"] == 0) & (merged["exact_weighted"] == 1)].copy()
    lost = merged[(merged["exact_gate27"] == 1) & (merged["exact_weighted"] == 0)].copy()

    print("gate27 score:", int(merged["exact_gate27"].sum()))
    print("weighted score:", int(merged["exact_weighted"].sum()))
    print("\nGained by weighted:")
    print(gained[["race_id", "track", "total_laps", "track_temp", "pred_top5_gate27", "pred_top5_weighted"]].to_string(index=False))
    print("\nLost by weighted:")
    print(lost[["race_id", "track", "total_laps", "track_temp", "pred_top5_gate27", "pred_top5_weighted"]].to_string(index=False))


if __name__ == "__main__":
    main()
