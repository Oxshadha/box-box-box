#!/usr/bin/env python3
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HISTORICAL_DIR = ROOT / "data" / "historical_races"
OUTPUT_CSV = ROOT / "analysis" / "historical_driver_features.csv"

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


def build_stints(strategy, total_laps):
    tire = strategy["starting_tire"]
    previous_lap = 0
    stints = []

    for stop in strategy["pit_stops"]:
        stop_lap = stop["lap"]
        stints.append((tire, stop_lap - previous_lap))
        tire = stop["to_tire"]
        previous_lap = stop_lap

    stints.append((tire, total_laps - previous_lap))
    return stints


def sequence_from_stints(stints):
    return ">".join(compound for compound, _ in stints)


def feature_row(race, grid_slot, strategy, finish_rank):
    race_config = race["race_config"]
    stints = build_stints(strategy, race_config["total_laps"])
    first_stop_lap = strategy["pit_stops"][0]["lap"] if strategy["pit_stops"] else ""
    second_stop_lap = strategy["pit_stops"][1]["lap"] if len(strategy["pit_stops"]) > 1 else ""

    row = {
        "race_id": race["race_id"],
        "grid_slot": grid_slot,
        "driver_id": strategy["driver_id"],
        "track": race_config["track"],
        "total_laps": race_config["total_laps"],
        "base_lap_time": race_config["base_lap_time"],
        "pit_lane_time": race_config["pit_lane_time"],
        "track_temp": race_config["track_temp"],
        "starting_tire": strategy["starting_tire"],
        "pit_stop_count": len(strategy["pit_stops"]),
        "first_stop_lap": first_stop_lap,
        "second_stop_lap": second_stop_lap,
        "tire_sequence": sequence_from_stints(stints),
        "stint_count": len(stints),
        "finish_rank": finish_rank,
        "finish_rank_pct": (finish_rank - 1) / 19.0,
    }

    for compound in COMPOUNDS:
        compound_stints = [length for tire, length in stints if tire == compound]
        row[f"{compound.lower()}_laps"] = sum(compound_stints)
        row[f"{compound.lower()}_stints"] = len(compound_stints)
        row[f"{compound.lower()}_max_stint"] = max(compound_stints, default=0)

    for index in range(3):
        row[f"stint_{index + 1}_compound"] = stints[index][0] if index < len(stints) else ""
        row[f"stint_{index + 1}_laps"] = stints[index][1] if index < len(stints) else 0

    return row


def main():
    rows = []
    for path in sorted(HISTORICAL_DIR.glob("races_*.json")):
        races = json.loads(path.read_text())
        for race in races:
            finish_lookup = {
                driver_id: rank
                for rank, driver_id in enumerate(race["finishing_positions"], start=1)
            }
            for grid_slot, strategy in sorted(race["strategies"].items()):
                rows.append(
                    feature_row(
                        race,
                        grid_slot,
                        strategy,
                        finish_lookup[strategy["driver_id"]],
                    )
                )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
