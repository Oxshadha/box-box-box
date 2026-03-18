#!/usr/bin/env python3
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

import train_xgb_ranker as base


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "analysis" / "models" / "rule_program_search"
PROGRAM_PATH = OUTPUT_DIR / "program.json"
METRICS_PATH = OUTPUT_DIR / "metrics.json"


def top_count_bucket(top_count):
    if top_count <= 5:
        return "4_5"
    if top_count <= 7:
        return "6_7"
    return "8_10"


def unique_bucket(unique_sequences):
    return str(min(int(unique_sequences), 8))


def build_template(frame, include_temp=True, include_unique=True):
    vc = frame["tire_sequence"].value_counts()
    parts = [str(frame["lap_band"].iloc[0])]
    if include_temp:
        parts.append(str(frame["temp_band"].iloc[0]))
    parts.extend(
        [
            str(vc.index[0]),
            top_count_bucket(int(vc.iloc[0])),
            f"s{int((frame['pit_stop_count'] == 2).sum())}",
        ]
    )
    if include_unique:
        parts.append(f"u{unique_bucket(frame['tire_sequence'].nunique())}")
    return "|".join(parts)


def template_keys(frame):
    return [
        build_template(frame, include_temp=True, include_unique=True),
        build_template(frame, include_temp=True, include_unique=False),
        build_template(frame, include_temp=False, include_unique=True),
        build_template(frame, include_temp=False, include_unique=False),
    ]


def compile_template_program(frame):
    seq_pair_stats = defaultdict(lambda: {"wins": 0.0, "total": 0.0})
    stop_pref_stats = defaultdict(lambda: {"later": 0.0, "earlier": 0.0, "total": 0.0})

    for _, race in frame.groupby("race_id", sort=False):
        race = race.sort_values(["finish_rank", "driver_id"], ascending=[True, True]).reset_index(drop=True)
        n = len(race)
        for i in range(n - 1):
            for j in range(i + 1, n):
                left = race.iloc[i]
                right = race.iloc[j]
                if left["tire_sequence"] != right["tire_sequence"]:
                    key = tuple(sorted((str(left["tire_sequence"]), str(right["tire_sequence"]))))
                    preferred = str(left["tire_sequence"])
                    seq_pair_stats[key]["wins"] += 1.0 if preferred == key[0] else 0.0
                    seq_pair_stats[key]["total"] += 1.0
                else:
                    stop_a = float(left["first_stop_lap"])
                    stop_b = float(right["first_stop_lap"])
                    if stop_a == stop_b:
                        continue
                    seq = str(left["tire_sequence"])
                    later_first = stop_a > stop_b
                    stop_pref_stats[seq]["later"] += 1.0 if later_first else 0.0
                    stop_pref_stats[seq]["earlier"] += 1.0 if not later_first else 0.0
                    stop_pref_stats[seq]["total"] += 1.0

    precedence_scores = defaultdict(float)
    pair_rules = {}
    for key, stats in seq_pair_stats.items():
        if stats["total"] < 8:
            continue
        win_rate_first = stats["wins"] / stats["total"]
        if abs(win_rate_first - 0.5) < 0.1:
            continue
        preferred = key[0] if win_rate_first > 0.5 else key[1]
        pair_rules["|".join(key)] = {
            "preferred": preferred,
            "count": int(stats["total"]),
            "win_rate": float(max(win_rate_first, 1.0 - win_rate_first)),
        }
        loser = key[1] if preferred == key[0] else key[0]
        precedence_scores[preferred] += 1.0
        precedence_scores[loser] -= 1.0

    stop_rules = {}
    for seq, stats in stop_pref_stats.items():
        if stats["total"] < 8:
            continue
        later_rate = stats["later"] / stats["total"]
        if abs(later_rate - 0.5) < 0.15:
            continue
        stop_rules[seq] = {
            "preferred": "later_first_stop" if later_rate > 0.5 else "earlier_first_stop",
            "count": int(stats["total"]),
            "win_rate": float(max(later_rate, 1.0 - later_rate)),
        }

    return {
        "sequence_scores": dict(precedence_scores),
        "pair_rules": pair_rules,
        "stop_rules": stop_rules,
    }


def learn_program(train):
    program = {}
    exact_templates = defaultdict(list)
    for _, race in train.groupby("race_id", sort=False):
        exact_templates[build_template(race, include_temp=True, include_unique=True)].append(race["race_id"].iloc[0])

    for key, race_ids in exact_templates.items():
        sub = train[train["race_id"].isin(race_ids)].copy()
        if sub["race_id"].nunique() < 8:
            continue
        program[key] = compile_template_program(sub)

    # backoff templates
    backoff_maps = [
        ("temp", lambda r: build_template(r, include_temp=True, include_unique=False)),
        ("unique", lambda r: build_template(r, include_temp=False, include_unique=True)),
        ("coarse", lambda r: build_template(r, include_temp=False, include_unique=False)),
    ]
    for _, key_fn in backoff_maps:
        grouped = defaultdict(list)
        for _, race in train.groupby("race_id", sort=False):
            grouped[key_fn(race)].append(race["race_id"].iloc[0])
        for key, race_ids in grouped.items():
            if key in program:
                continue
            sub = train[train["race_id"].isin(race_ids)].copy()
            if sub["race_id"].nunique() < 12:
                continue
            program[key] = compile_template_program(sub)
    return program


def rank_race(race, program):
    race = race.copy()
    key_candidates = template_keys(race)
    chosen = None
    for key in key_candidates:
        if key in program:
            chosen = program[key]
            break
    if chosen is None:
        chosen = {"sequence_scores": {}, "pair_rules": {}, "stop_rules": {}}

    sequence_scores = chosen.get("sequence_scores", {})
    pair_rules = chosen.get("pair_rules", {})
    stop_rules = chosen.get("stop_rules", {})

    race["sequence_score"] = race["tire_sequence"].map(sequence_scores).fillna(0.0).astype(float)
    race["stop_order_score"] = 0.0
    for seq, rule in stop_rules.items():
        mask = race["tire_sequence"] == seq
        if not mask.any():
            continue
        values = race.loc[mask, "first_stop_lap"].astype(float)
        if rule["preferred"] == "later_first_stop":
            race.loc[mask, "stop_order_score"] = values
        else:
            race.loc[mask, "stop_order_score"] = -values

    race = race.sort_values(
        ["sequence_score", "stop_order_score", "pred_blend", "driver_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    changed = True
    while changed:
        changed = False
        for i in range(len(race) - 1):
            left = race.iloc[i]
            right = race.iloc[i + 1]
            pair_key = "|".join(sorted((str(left["tire_sequence"]), str(right["tire_sequence"]))))
            rule = pair_rules.get(pair_key)
            if not rule:
                continue
            preferred = str(rule["preferred"])
            if str(right["tire_sequence"]) == preferred and str(left["tire_sequence"]) != preferred:
                race.iloc[[i, i + 1]] = race.iloc[[i + 1, i]].to_numpy()
                changed = True
    return race


def evaluate(frame, program):
    exact = 0
    top3 = 0
    top5 = 0
    kendalls = []
    spearmans = []
    races = 0
    for _, race in frame.groupby("race_id", sort=False):
        ranked = rank_race(race, program)
        pred_ids = ranked["driver_id"].tolist()
        actual = race.sort_values(["finish_rank", "driver_id"], ascending=[True, True])
        actual_ids = actual["driver_id"].tolist()
        exact += int(pred_ids == actual_ids)
        top3 += int(set(pred_ids[:3]) == set(actual_ids[:3]))
        top5 += int(set(pred_ids[:5]) == set(actual_ids[:5]))
        merged = actual[["driver_id", "finish_rank"]].merge(
            ranked[["driver_id"]].assign(pred_rank=np.arange(1, len(ranked) + 1)),
            on="driver_id",
            how="inner",
        )
        kendalls.append(float(kendalltau(merged["finish_rank"], merged["pred_rank"]).statistic))
        spearmans.append(float(spearmanr(merged["finish_rank"], merged["pred_rank"]).statistic))
        races += 1
    return {
        "races": races,
        "exact_order_accuracy": exact / races if races else 0.0,
        "top3_set_accuracy": top3 / races if races else 0.0,
        "top5_set_accuracy": top5 / races if races else 0.0,
        "mean_kendall_tau": float(np.mean(kendalls)) if kendalls else 0.0,
        "mean_spearman_rho": float(np.mean(spearmans)) if spearmans else 0.0,
    }


def main():
    df, splits, v2_record, v4_record = base.load_inputs()
    df = base.add_physics_predictions(df, v2_record, v4_record)
    df = base.add_regime_features(df)

    train = df[df["race_id"].isin(splits["train"])].copy()
    validation = df[df["race_id"].isin(splits["validation"])].copy()
    test = df[df["race_id"].isin(splits["test"])].copy()

    program = learn_program(train)
    metrics = {
        "validation": {"rule_program": evaluate(validation, program)},
        "test": {"rule_program": evaluate(test, program)},
        "program": {
            "templates": int(len(program)),
            "templates_with_pair_rules": int(sum(bool(v.get("pair_rules")) for v in program.values())),
            "templates_with_stop_rules": int(sum(bool(v.get("stop_rules")) for v in program.values())),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRAM_PATH.write_text(json.dumps(program, indent=2))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
