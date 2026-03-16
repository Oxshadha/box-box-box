#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "analysis" / "solver_snapshots"


def load_snapshot(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.is_absolute():
        path = OUTPUT_DIR / path
    return pd.DataFrame(json.loads(path.read_text()))


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: compare_solver_snapshots.py SNAPSHOT_A SNAPSHOT_B")
    a_name = sys.argv[1]
    b_name = sys.argv[2]
    a = load_snapshot(a_name).rename(columns={"exact_match": "exact_a", "pred_top5": "pred_top5_a"})
    b = load_snapshot(b_name).rename(columns={"exact_match": "exact_b", "pred_top5": "pred_top5_b"})
    merged = a.merge(b[["race_id", "exact_b", "pred_top5_b"]], on="race_id", how="inner")
    gained = merged[(merged["exact_a"] == 0) & (merged["exact_b"] == 1)].copy()
    lost = merged[(merged["exact_a"] == 1) & (merged["exact_b"] == 0)].copy()
    both = merged[(merged["exact_a"] == 1) & (merged["exact_b"] == 1)].copy()

    print("A score:", int(merged["exact_a"].sum()))
    print("B score:", int(merged["exact_b"].sum()))
    print("Shared wins:", int(len(both)))
    print("\nGained by B:")
    print(gained[["race_id", "track", "total_laps", "track_temp", "pred_top5_a", "pred_top5_b"]].to_string(index=False))
    print("\nLost by B:")
    print(lost[["race_id", "track", "total_laps", "track_temp", "pred_top5_a", "pred_top5_b"]].to_string(index=False))


if __name__ == "__main__":
    main()
