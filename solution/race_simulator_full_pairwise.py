#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = ROOT / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import full_pairwise_model_lib as lib  # noqa: E402
import race_simulator_xgb_gate as gate  # noqa: E402


PAIRWISE_MODEL, _ = lib.load_model_bundle()


def predict_finishing_positions(test_case):
    rows = []
    for grid_slot in sorted(test_case["strategies"]):
        strategy = test_case["strategies"][grid_slot]
        row = gate.feature_row(strategy, test_case["race_config"])
        row["driver_id"] = strategy["driver_id"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    ranked = lib.rank_race_with_pairwise_model(frame, PAIRWISE_MODEL)
    return ranked["driver_id"].tolist()


def main():
    test_case = json.load(sys.stdin)
    output = {
        "race_id": test_case["race_id"],
        "finishing_positions": predict_finishing_positions(test_case),
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
