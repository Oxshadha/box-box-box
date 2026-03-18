#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "test_cases" / "inputs"
SNAPSHOT_PATH = ROOT / "analysis" / "solver_snapshots" / "gate33.json"
OUTPUT_DIR = ROOT / "analysis" / "gate33_failure_families"


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


def top_count_bucket(top_count: int) -> str:
    if top_count <= 5:
        return "4_5"
    if top_count <= 7:
        return "6_7"
    return "8_10"


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
    tb = temp_band(track_temp)

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
    dominant_soft_hard = top_count >= 6 and top_sequence == "SOFT>HARD"
    dominant_hard_medium_short = top_count >= 6 and top_sequence == "HARD>MEDIUM" and total_laps <= 35
    dominant_medium_hard_short = top_count >= 6 and top_sequence == "MEDIUM>HARD" and total_laps <= 35
    if total_laps <= 35 and unique_sequences <= 4:
        return "no_comp"
    if dominant_soft_hard and two_stop == 0 and unique_sequences <= 6 and total_laps <= 55:
        return "no_comp"
    if dominant_hard_medium_short and two_stop == 0 and unique_sequences <= 5 and tb in {"mild", "warm"}:
        return "no_comp"
    if dominant_medium_hard_short and two_stop == 0 and unique_sequences <= 5:
        return "no_comp"
    return "comp"


def sequence_signature_bucket(counts: Counter) -> str:
    items = counts.most_common(3)
    return " | ".join(f"{seq}:{count}" for seq, count in items)


def main():
    snapshot = pd.DataFrame(json.loads(SNAPSHOT_PATH.read_text()))
    snapshot = snapshot.rename(columns={"exact_match": "passed"})
    failed_ids = set(snapshot.loc[snapshot["passed"] == 0, "race_id"])

    rows = []
    for input_path in sorted(INPUT_DIR.glob("test_*.json")):
        race = json.loads(input_path.read_text())
        race_id = race["race_id"]
        if race_id not in failed_ids:
            continue
        row = snapshot[snapshot["race_id"] == race_id].iloc[0]
        strategies = list(race["strategies"].values())
        sequences = [
            ">".join([s["starting_tire"]] + [stop["to_tire"] for stop in s["pit_stops"]])
            for s in strategies
        ]
        counts = Counter(sequences)
        top_sequence, top_count = counts.most_common(1)[0]
        total_laps = int(race["race_config"]["total_laps"])
        track_temp = float(race["race_config"]["track_temp"])
        two_stop = sum(1 for s in strategies if len(s["pit_stops"]) == 2)
        unique_sequences = len(counts)
        lb = lap_band(total_laps)
        tb = temp_band(track_temp)
        branch = route_branch(race)
        top5_correct = int(row["expected_top5"] == row["pred_top5"])
        family_key = "|".join(
            [
                branch,
                "top5ok" if top5_correct else "top5miss",
                lb,
                tb,
                top_sequence,
                top_count_bucket(int(top_count)),
                f"u{unique_sequences}",
                f"s{two_stop}",
            ]
        )
        rows.append(
            {
                "race_id": race_id,
                "family_key": family_key,
                "branch": branch,
                "top5_correct": top5_correct,
                "track": race["race_config"]["track"],
                "total_laps": total_laps,
                "track_temp": track_temp,
                "lap_band": lb,
                "temp_band": tb,
                "top_sequence": top_sequence,
                "top_count": int(top_count),
                "top_count_bucket": top_count_bucket(int(top_count)),
                "unique_sequences": unique_sequences,
                "two_stop_drivers": two_stop,
                "signature_top3": sequence_signature_bucket(counts),
            }
        )

    failed = pd.DataFrame(rows)

    family_summary = (
        failed.groupby(
            [
                "family_key",
                "branch",
                "top5_correct",
                "lap_band",
                "temp_band",
                "top_sequence",
                "top_count_bucket",
                "unique_sequences",
                "two_stop_drivers",
            ],
            dropna=False,
        )
        .agg(
            races=("race_id", "count"),
            tracks=("track", lambda s: ",".join(sorted(set(s)))),
            mean_laps=("total_laps", "mean"),
            mean_temp=("track_temp", "mean"),
            sample_cases=("race_id", lambda s: ",".join(list(s.head(5)))),
        )
        .reset_index()
        .sort_values(
            ["races", "top5_correct", "branch", "lap_band", "temp_band", "top_sequence", "family_key"],
            ascending=[False, False, True, True, True, True, True],
        )
    )

    summary = {
        "failed_cases": int(len(failed)),
        "top5_correct_failed_cases": int(failed["top5_correct"].sum()),
        "family_count": int(len(family_summary)),
        "largest_families": family_summary.head(25).round(3).to_dict(orient="records"),
        "branch_split": failed.groupby(["branch", "top5_correct"]).size().reset_index(name="races").to_dict(orient="records"),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    family_summary.round(3).to_csv(OUTPUT_DIR / "family_summary.csv", index=False)
    failed.to_csv(OUTPUT_DIR / "failed_cases.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
