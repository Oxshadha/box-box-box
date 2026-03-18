#!/usr/bin/env python3
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "test_cases" / "inputs"
EXPECTED_DIR = ROOT / "data" / "test_cases" / "expected_outputs"
SNAPSHOT_PATH = ROOT / "analysis" / "solver_snapshots" / "gate33.json"
DEFAULT_SOLVER = ROOT / "solution" / "race_simulator_xgb_gate.py"
OUTPUT_DIR = ROOT / "analysis" / "reverse_engineering_gate33"

TOP_KEEP = 5


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


def strategy_rows(race: dict):
    rows = []
    for grid_slot in sorted(race["strategies"]):
        strategy = race["strategies"][grid_slot]
        total_laps = float(race["race_config"]["total_laps"])
        first_stop = float(strategy["pit_stops"][0]["lap"] if strategy["pit_stops"] else 0.0)
        second_stop = float(strategy["pit_stops"][1]["lap"] if len(strategy["pit_stops"]) > 1 else 0.0)
        tire_sequence = ">".join([strategy["starting_tire"]] + [stop["to_tire"] for stop in strategy["pit_stops"]])
        rows.append(
            {
                "driver_id": strategy["driver_id"],
                "grid_slot": grid_slot,
                "starting_tire": strategy["starting_tire"],
                "pit_stop_count": len(strategy["pit_stops"]),
                "first_stop_frac": first_stop / total_laps,
                "second_stop_frac": second_stop / total_laps,
                "tire_sequence": tire_sequence,
            }
        )
    return pd.DataFrame(rows)


def template_key(race: dict) -> str:
    strategies = list(race["strategies"].values())
    sequences = [
        ">".join([s["starting_tire"]] + [stop["to_tire"] for stop in s["pit_stops"]])
        for s in strategies
    ]
    counts = Counter(sequences)
    top_sequence, top_count = counts.most_common(1)[0]
    return "|".join(
        [
            lap_band(int(race["race_config"]["total_laps"])),
            temp_band(float(race["race_config"]["track_temp"])),
            top_sequence,
            top_count_bucket(int(top_count)),
        ]
    )


def branch_name(race: dict) -> str:
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


def seq_pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def main():
    solver_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_SOLVER
    snapshot = pd.DataFrame(json.loads(SNAPSHOT_PATH.read_text()))
    failed_ids = set(snapshot.loc[snapshot["exact_match"] == 0, "race_id"])

    template_pair_stats = defaultdict(lambda: {"count": 0, "wins": Counter()})
    within_seq_rules = defaultdict(lambda: {"count": 0, "earlier_first_stop_better": 0, "later_first_stop_better": 0})
    case_rows = []

    for input_path in sorted(INPUT_DIR.glob("test_*.json")):
        race = json.loads(input_path.read_text())
        race_id = race["race_id"]
        if race_id not in failed_ids:
            continue
        expected = json.loads((EXPECTED_DIR / input_path.name).read_text())["finishing_positions"]
        predicted = run_solver(solver_path, input_path)["finishing_positions"]
        info = strategy_rows(race)
        info["expected_rank"] = info["driver_id"].map({d: i + 1 for i, d in enumerate(expected)})
        info["pred_rank"] = info["driver_id"].map({d: i + 1 for i, d in enumerate(predicted)})

        key = template_key(race)
        branch = branch_name(race)
        top5_correct = expected[:5] == predicted[:5]
        tail = info[(info["expected_rank"] > TOP_KEEP) | (info["pred_rank"] > TOP_KEEP)].copy()
        tail = tail.sort_values(["pred_rank", "driver_id"], ascending=[True, True]).reset_index(drop=True)

        for i in range(len(tail) - 1):
            left = tail.iloc[i]
            right = tail.iloc[i + 1]
            if left["tire_sequence"] == right["tire_sequence"]:
                continue
            winner = left["tire_sequence"] if int(left["expected_rank"]) < int(right["expected_rank"]) else right["tire_sequence"]
            stat = template_pair_stats[(branch, key, seq_pair_key(left["tire_sequence"], right["tire_sequence"]))]
            stat["count"] += 1
            stat["wins"][winner] += 1

        for seq_name, seq_group in tail.groupby("tire_sequence"):
            if len(seq_group) < 2:
                continue
            seq_group = seq_group.sort_values(["pred_rank", "driver_id"], ascending=[True, True]).reset_index(drop=True)
            for i in range(len(seq_group) - 1):
                left = seq_group.iloc[i]
                right = seq_group.iloc[i + 1]
                if float(left["first_stop_frac"]) == float(right["first_stop_frac"]):
                    continue
                rule = within_seq_rules[(branch, key, seq_name)]
                rule["count"] += 1
                earlier_is_left = float(left["first_stop_frac"]) < float(right["first_stop_frac"])
                left_better = int(left["expected_rank"]) < int(right["expected_rank"])
                if earlier_is_left == left_better:
                    rule["earlier_first_stop_better"] += 1
                else:
                    rule["later_first_stop_better"] += 1

        case_rows.append(
            {
                "race_id": race_id,
                "branch": branch,
                "template": key,
                "track": race["race_config"]["track"],
                "total_laps": int(race["race_config"]["total_laps"]),
                "track_temp": float(race["race_config"]["track_temp"]),
                "top5_correct": int(top5_correct),
            }
        )

    pair_rules = []
    for (branch, key, pair_key), stat in template_pair_stats.items():
        if stat["count"] < 3:
            continue
        preferred, wins = stat["wins"].most_common(1)[0]
        win_rate = wins / stat["count"]
        if win_rate < 0.7:
            continue
        pair_rules.append(
            {
                "branch": branch,
                "template": key,
                "pair": pair_key,
                "preferred": preferred,
                "count": int(stat["count"]),
                "win_rate": float(win_rate),
            }
        )

    stop_rules = []
    for (branch, key, seq_name), stat in within_seq_rules.items():
        if stat["count"] < 3:
            continue
        earlier_rate = stat["earlier_first_stop_better"] / stat["count"]
        later_rate = stat["later_first_stop_better"] / stat["count"]
        if max(earlier_rate, later_rate) < 0.7:
            continue
        stop_rules.append(
            {
                "branch": branch,
                "template": key,
                "sequence": seq_name,
                "count": int(stat["count"]),
                "preferred": "earlier_first_stop" if earlier_rate >= later_rate else "later_first_stop",
                "win_rate": float(max(earlier_rate, later_rate)),
            }
        )

    summary = {
        "failed_cases": len(case_rows),
        "top5_correct_failed_cases": int(sum(row["top5_correct"] for row in case_rows)),
        "pair_rule_candidates": sorted(pair_rules, key=lambda x: (-x["count"], -x["win_rate"], x["template"], x["pair"]))[:80],
        "within_sequence_stop_rules": sorted(stop_rules, key=lambda x: (-x["count"], -x["win_rate"], x["template"], x["sequence"]))[:80],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(case_rows).to_csv(OUTPUT_DIR / "failed_cases.csv", index=False)
    pd.DataFrame(pair_rules).to_csv(OUTPUT_DIR / "pair_rule_candidates.csv", index=False)
    pd.DataFrame(stop_rules).to_csv(OUTPUT_DIR / "within_sequence_stop_rules.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
