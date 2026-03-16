#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "test_cases" / "inputs"
EXPECTED_DIR = ROOT / "data" / "test_cases" / "expected_outputs"
DEFAULT_SOLVER = ROOT / "solution" / "race_simulator_xgb.py"
OUTPUT_DIR = ROOT / "analysis" / "solver_snapshots"


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


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "snapshot"
    solver = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else DEFAULT_SOLVER
    rows = []
    for input_path in sorted(INPUT_DIR.glob("test_*.json")):
        race = json.loads(input_path.read_text())
        expected = json.loads((EXPECTED_DIR / input_path.name).read_text())
        pred = run_solver(solver, input_path)
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{name}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(out)


if __name__ == "__main__":
    main()
