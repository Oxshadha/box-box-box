#!/usr/bin/env python3
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from xgboost import XGBRanker

import train_xgb_ranker as base


ROOT = Path(__file__).resolve().parent.parent
COMP_DIR = ROOT / "analysis" / "models" / "xgb_ranker_comp"
OUTPUT_DIR = ROOT / "analysis" / "models" / "comp_tail_rules"
RULES_PATH = OUTPUT_DIR / "rules.json"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

TOP_KEEP = 5
PAIR_GAP = 0.22
MIN_COUNT = 12
MIN_WIN_RATE = 0.72


def load_comp_bundle():
    metadata = json.loads((COMP_DIR / "metadata.json").read_text())
    ranker = XGBRanker()
    ranker.load_model(str(COMP_DIR / "model.json"))
    return metadata, ranker


def encode_with_metadata(frame, metadata):
    frame = frame.copy()
    for col in metadata["categorical_columns"]:
        frame[col] = frame[col].fillna("__MISSING__").astype(str)
    numeric_cols = [c for c in metadata["feature_columns"] if c not in metadata["categorical_columns"]]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame = pd.get_dummies(frame, columns=metadata["categorical_columns"], dummy_na=True)
    for col in metadata["encoded_columns"]:
        if col not in frame.columns:
            frame[col] = 0
    return frame[metadata["encoded_columns"]]


def top_count_bucket(top_count: int) -> str:
    if top_count <= 5:
        return "4_5"
    if top_count <= 7:
        return "6_7"
    return "8_10"


def route_comp(frame):
    race_level = (
        frame.groupby("race_id", sort=False)
        .agg(
            total_laps=("total_laps", "first"),
            track_temp=("track_temp", "first"),
            temp_band=("temp_band", "first"),
            lap_band=("lap_band", "first"),
            unique_sequences=("tire_sequence", "nunique"),
            two_stop=("pit_stop_count", lambda s: int((s == 2).sum())),
            top_sequence=("tire_sequence", lambda s: str(s.value_counts().index[0])),
            top_count=("tire_sequence", lambda s: int(s.value_counts().iloc[0])),
        )
        .reset_index()
    )

    dominant_soft_hard = (race_level["top_count"] >= 6) & (race_level["top_sequence"] == "SOFT>HARD")
    dominant_hard_medium_short = (
        (race_level["top_count"] >= 6)
        & (race_level["top_sequence"] == "HARD>MEDIUM")
        & (race_level["total_laps"] <= 35)
    )
    dominant_medium_hard_short = (
        (race_level["top_count"] >= 6)
        & (race_level["top_sequence"] == "MEDIUM>HARD")
        & (race_level["total_laps"] <= 35)
    )
    balanced_cluster = (
        (race_level["two_stop"] == 0)
        & race_level["unique_sequences"].between(4, 6)
        & race_level["track_temp"].between(28, 36)
        & (race_level["total_laps"] <= 45)
        & race_level["top_count"].between(4, 10)
        & race_level["top_sequence"].isin(["MEDIUM>HARD", "SOFT>HARD", "SOFT>MEDIUM", "MEDIUM>SOFT"])
    )

    no_comp = (
        ((race_level["total_laps"] <= 35) & (race_level["unique_sequences"] <= 4))
        | (
            dominant_soft_hard
            & (race_level["two_stop"] == 0)
            & (race_level["unique_sequences"] <= 6)
            & (race_level["total_laps"] <= 55)
        )
        | (
            dominant_hard_medium_short
            & (race_level["two_stop"] == 0)
            & (race_level["unique_sequences"] <= 5)
            & race_level["temp_band"].isin(["mild", "warm"])
        )
        | (
            dominant_medium_hard_short
            & (race_level["two_stop"] == 0)
            & (race_level["unique_sequences"] <= 5)
        )
    )
    keep = set(race_level.loc[~balanced_cluster & ~no_comp, "race_id"])
    return frame[frame["race_id"].isin(keep)].copy()


def template_key(race: pd.DataFrame) -> str:
    top_counts = race["tire_sequence"].value_counts()
    top_sequence = str(top_counts.index[0])
    top_count = int(top_counts.iloc[0])
    return "|".join(
        [
            str(race["lap_band"].iloc[0]),
            str(race["temp_band"].iloc[0]),
            top_sequence,
            top_count_bucket(top_count),
        ]
    )


def mine_rules(frame: pd.DataFrame):
    pair_stats = defaultdict(lambda: {"count": 0, "wins": Counter()})
    template_counts = Counter()
    for _, race in frame.groupby("race_id", sort=False):
        race = race.sort_values(["base_score", "driver_id"], ascending=[True, True]).reset_index(drop=True)
        if len(race) <= TOP_KEEP:
            continue
        key = template_key(race)
        template_counts[key] += 1
        tail = race.iloc[TOP_KEEP:].reset_index(drop=True)
        for i in range(len(tail) - 1):
            left = tail.iloc[i]
            right = tail.iloc[i + 1]
            if abs(float(left["base_score"] - right["base_score"])) > PAIR_GAP:
                continue
            if left["tire_sequence"] == right["tire_sequence"]:
                continue
            pair_key = tuple(sorted((str(left["tire_sequence"]), str(right["tire_sequence"]))))
            winner = str(left["tire_sequence"]) if int(left["finish_rank"]) < int(right["finish_rank"]) else str(right["tire_sequence"])
            slot = pair_stats[(key, pair_key)]
            slot["count"] += 1
            slot["wins"][winner] += 1

    rules = {}
    for (key, pair_key), stats in pair_stats.items():
        count = int(stats["count"])
        if count < MIN_COUNT:
            continue
        preferred, wins = stats["wins"].most_common(1)[0]
        win_rate = wins / count
        if win_rate < MIN_WIN_RATE:
            continue
        rules.setdefault(key, {})
        rules[key]["|".join(pair_key)] = {
            "preferred": preferred,
            "count": count,
            "win_rate": win_rate,
        }
    return rules, template_counts


def main():
    df, splits, v2_record, v4_record = base.load_inputs()
    df = base.add_physics_predictions(df, v2_record, v4_record)
    df = base.add_regime_features(df)
    metadata, ranker = load_comp_bundle()
    df = base.add_race_composition_features(df, metadata.get("major_sequences", []))
    encoded = encode_with_metadata(df[metadata["feature_columns"]], metadata)
    df["base_score"] = ranker.predict(encoded)
    comp = route_comp(df)

    train = comp[comp["race_id"].isin(splits["train"])].copy()
    validation = comp[comp["race_id"].isin(splits["validation"])].copy()
    rules, template_counts = mine_rules(train)

    summary = {
        "rule_templates": len(rules),
        "templates_with_rules": sorted(
            [{"template": key, "rules": len(value), "train_races": int(template_counts.get(key, 0))} for key, value in rules.items()],
            key=lambda item: (-item["rules"], item["template"]),
        ),
        "validation_comp_races": int(validation["race_id"].nunique()),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RULES_PATH.write_text(json.dumps(rules, indent=2))
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
