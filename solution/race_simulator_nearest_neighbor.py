#!/usr/bin/env python3
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

from nearest_neighbor_solver_lib import load_artifact, nearest_order  # noqa: E402


MODEL_DIR = ROOT / "analysis" / "models" / "nearest_neighbor"
ARTIFACT = load_artifact(MODEL_DIR)


def main():
    test_case = json.load(sys.stdin)
    output = {
        "race_id": test_case["race_id"],
        "finishing_positions": nearest_order(test_case, ARTIFACT),
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
