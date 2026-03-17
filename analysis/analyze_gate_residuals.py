#!/usr/bin/env python3
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "test_cases" / "inputs"
SNAPSHOT_DIR = ROOT / "analysis" / "solver_snapshots"
OUTPUT_DIR = ROOT / "analysis" / "gate_residuals"


def load_snapshot(name: str) -> pd.DataFrame:
    path = Path(name)
    if not path.is_absolute():
        path = SNAPSHOT_DIR / path
    return pd.DataFrame(json.loads(path.read_text()))


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


def choose_branch(race: dict) -> str:
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
    lb = lap_band(total_laps)
    tb = temp_band(track_temp)

    dominant_soft_hard = top_count >= 6 and top_sequence == "SOFT>HARD"
    dominant_hard_medium_short = top_count >= 6 and top_sequence == "HARD>MEDIUM" and total_laps <= 35
    dominant_medium_hard_short = top_count >= 6 and top_sequence == "MEDIUM>HARD" and total_laps <= 35
    balanced_cluster = (
        two_stop == 0
        and 4 <= unique_sequences <= 6
        and 28 <= track_temp <= 36
        and total_laps <= 45
        and 4 <= top_count <= 10
    )

    if balanced_cluster and top_sequence in {"MEDIUM>HARD", "SOFT>HARD", "SOFT>MEDIUM", "MEDIUM>SOFT"}:
        return "balanced"
    if total_laps <= 35 and unique_sequences <= 4:
        return "no_comp"
    if dominant_soft_hard and two_stop == 0 and unique_sequences <= 6 and total_laps <= 55:
        return "no_comp"
    if dominant_hard_medium_short and two_stop == 0 and unique_sequences <= 5 and tb in {"mild", "warm"}:
        return "no_comp"
    if dominant_medium_hard_short and two_stop == 0 and unique_sequences <= 5:
        return "no_comp"
    if two_stop >= 5:
        return "comp"
    if top_sequence == "HARD>MEDIUM" and lb in {"mid", "long"}:
        return "comp"
    if total_laps >= 48:
        return "comp"
    if unique_sequences >= 7:
        return "comp"
    return "comp"


def summarize_group(rows: pd.DataFrame, key: str) -> list[dict]:
    grouped = (
        rows.groupby(key)
        .agg(
            races=("race_id", "count"),
            mean_laps=("total_laps", "mean"),
            mean_temp=("track_temp", "mean"),
            mean_unique_sequences=("unique_sequence_count", "mean"),
            mean_top_count=("top_sequence_count", "mean"),
        )
        .reset_index()
        .sort_values(["races", key], ascending=[False, True])
    )
    return grouped.round(3).to_dict(orient="records")


def main():
    snapshot_name = sys.argv[1] if len(sys.argv) > 1 else "gate24.json"
    snapshot = load_snapshot(snapshot_name)

    rows = []
    for input_path in sorted(INPUT_DIR.glob("test_*.json")):
        race = json.loads(input_path.read_text())
        strategies = list(race["strategies"].values())
        sequences = [
            ">".join([s["starting_tire"]] + [stop["to_tire"] for stop in s["pit_stops"]])
            for s in strategies
        ]
        counts = Counter(sequences)
        top_sequence, top_count = counts.most_common(1)[0]
        total_laps = int(race["race_config"]["total_laps"])
        track_temp = float(race["race_config"]["track_temp"])
        rows.append(
            {
                "race_id": race["race_id"],
                "track": race["race_config"]["track"],
                "total_laps": total_laps,
                "track_temp": track_temp,
                "pit_lane_time": float(race["race_config"]["pit_lane_time"]),
                "lap_band": lap_band(total_laps),
                "temp_band": temp_band(track_temp),
                "unique_sequence_count": len(counts),
                "top_sequence": top_sequence,
                "top_sequence_count": top_count,
                "two_stop_drivers": sum(1 for s in strategies if len(s["pit_stops"]) == 2),
                "branch": choose_branch(race),
                "sequence_signature": dict(counts.most_common(5)),
            }
        )

    features = pd.DataFrame(rows)
    merged = features.merge(snapshot[["race_id", "exact_match"]], on="race_id", how="inner")
    failed = merged[merged["exact_match"] == 0].copy()

    summary = {
        "snapshot": snapshot_name,
        "overall": {
            "races": int(len(merged)),
            "passed": int(merged["exact_match"].sum()),
            "failed": int(len(failed)),
        },
        "failed_by_branch": summarize_group(failed, "branch"),
        "failed_by_track": summarize_group(failed, "track"),
        "failed_by_lap_band": summarize_group(failed, "lap_band"),
        "failed_by_temp_band": summarize_group(failed, "temp_band"),
        "failed_by_top_sequence": summarize_group(failed, "top_sequence"),
        "sample_failed_cases": failed[
            [
                "race_id",
                "track",
                "total_laps",
                "track_temp",
                "lap_band",
                "temp_band",
                "branch",
                "unique_sequence_count",
                "top_sequence",
                "top_sequence_count",
                "two_stop_drivers",
            ]
        ]
        .sort_values(["branch", "track", "total_laps", "race_id"])
        .head(30)
        .to_dict(orient="records"),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
