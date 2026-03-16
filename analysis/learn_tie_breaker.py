#!/usr/bin/env python3
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

import blend_physics_simulators as blend_mod


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "analysis" / "historical_driver_features.csv"
SPLIT_PATH = ROOT / "analysis" / "splits" / "race_id_splits.json"
V2_PARAMS_PATH = ROOT / "analysis" / "models" / "physics_simulator_v2" / "best_params.json"
V4_PARAMS_PATH = ROOT / "analysis" / "models" / "physics_simulator_v4" / "best_params.json"
OUTPUT_DIR = ROOT / "analysis" / "models" / "physics_blend_v2_v4_tiebreak"
RULES_PATH = OUTPUT_DIR / "tie_break_rules.json"
METRICS_PATH = OUTPUT_DIR / "metrics.json"


def load_all():
    df = pd.read_csv(DATA_PATH)
    splits = json.loads(SPLIT_PATH.read_text())
    v2 = json.loads(V2_PARAMS_PATH.read_text())
    v4 = json.loads(V4_PARAMS_PATH.read_text())
    return df, splits, v2, v4


def prepare_split_frames(df, splits, v2_record, v4_record):
    data = {}
    for split_name in ("train", "validation", "test"):
        part = df[df["race_id"].isin(splits[split_name])].copy()

        part_v2 = blend_mod.sim_v2.add_piecewise_statistics(part, v2_record["thresholds"])
        part["pred_v2"] = blend_mod.sim_v2.predict_total_time(part_v2, v2_record["params"])

        part_v4 = blend_mod.sim_v4.add_piecewise_statistics(
            part,
            v4_record["thresholds"],
            v4_record["cliff_thresholds"],
        )
        part["pred_v4"] = blend_mod.sim_v4.predict_total_time(part_v4, v4_record["params"])
        data[split_name] = part
    return data


def sequence_from_row(row):
    return row["tire_sequence"]


def learn_rules(train_frame: pd.DataFrame, close_gap: float, min_count: int = 80, min_win_rate: float = 0.58):
    pair_stats = {}

    for _, race in train_frame.groupby("race_id", sort=False):
        rows = race.to_dict("records")
        true_rank = {row["driver_id"]: row["finish_rank"] for row in rows}
        for left, right in itertools.combinations(rows, 2):
            score_gap = abs(left["pred_blend"] - right["pred_blend"])
            if score_gap > close_gap:
                continue

            seq_left = sequence_from_row(left)
            seq_right = sequence_from_row(right)
            if seq_left == seq_right:
                continue

            key = tuple(sorted((seq_left, seq_right)))
            if key not in pair_stats:
                pair_stats[key] = {"count": 0, key[0] + "_wins": 0}
            pair_stats[key]["count"] += 1

            winner = left["driver_id"] if true_rank[left["driver_id"]] < true_rank[right["driver_id"]] else right["driver_id"]
            winner_seq = seq_left if winner == left["driver_id"] else seq_right
            if winner_seq == key[0]:
                pair_stats[key][key[0] + "_wins"] += 1

    rules = {}
    for key, stats in pair_stats.items():
        count = stats["count"]
        if count < min_count:
            continue
        win_rate = stats[key[0] + "_wins"] / count
        if win_rate >= min_win_rate:
            rules["|".join(key)] = {"preferred": key[0], "count": count, "win_rate": win_rate}
        elif win_rate <= 1.0 - min_win_rate:
            rules["|".join(key)] = {"preferred": key[1], "count": count, "win_rate": 1.0 - win_rate}
    return rules


def apply_rules(race: pd.DataFrame, close_gap: float, rules: dict):
    rows = race.sort_values(["pred_blend", "driver_id"], ascending=[True, True]).to_dict("records")
    changed = True
    while changed:
        changed = False
        for idx in range(len(rows) - 1):
            a = rows[idx]
            b = rows[idx + 1]
            if abs(a["pred_blend"] - b["pred_blend"]) > close_gap:
                continue
            key = "|".join(sorted((a["tire_sequence"], b["tire_sequence"])))
            rule = rules.get(key)
            if not rule:
                continue
            preferred = rule["preferred"]
            if b["tire_sequence"] == preferred and a["tire_sequence"] != preferred:
                rows[idx], rows[idx + 1] = rows[idx + 1], rows[idx]
                changed = True
    return pd.DataFrame(rows)


def evaluate(frame: pd.DataFrame, close_gap: float, rules: dict):
    exact = 0
    top3 = 0
    top5 = 0
    kendalls = []
    spearmans = []

    for _, race in frame.groupby("race_id", sort=False):
        predicted = apply_rules(race, close_gap, rules)
        actual = race.sort_values(["finish_rank", "driver_id"], ascending=[True, True])
        pred_ids = predicted["driver_id"].tolist()
        actual_ids = actual["driver_id"].tolist()

        exact += int(pred_ids == actual_ids)
        top3 += int(set(pred_ids[:3]) == set(actual_ids[:3]))
        top5 += int(set(pred_ids[:5]) == set(actual_ids[:5]))

        merged = actual[["driver_id", "finish_rank"]].merge(
            predicted[["driver_id"]].assign(pred_rank=np.arange(1, len(predicted) + 1)),
            on="driver_id",
            how="inner",
        )
        kendalls.append(float(kendalltau(merged["finish_rank"], merged["pred_rank"]).statistic))
        spearmans.append(float(spearmanr(merged["finish_rank"], merged["pred_rank"]).statistic))

    race_count = frame["race_id"].nunique()
    return {
        "races": int(race_count),
        "exact_order_accuracy": exact / race_count,
        "top3_set_accuracy": top3 / race_count,
        "top5_set_accuracy": top5 / race_count,
        "mean_kendall_tau": float(np.mean(kendalls)),
        "mean_spearman_rho": float(np.mean(spearmans)),
    }


def main():
    df, splits, v2_record, v4_record = load_all()
    split_frames = prepare_split_frames(df, splits, v2_record, v4_record)

    for split_name, frame in split_frames.items():
        split_frames[split_name] = frame.copy()
        split_frames[split_name]["pred_blend"] = 0.6 * frame["pred_v2"] + 0.4 * frame["pred_v4"]

    sampled_train_races = split_frames["train"]["race_id"].drop_duplicates().sample(n=200, random_state=42)
    sampled_validation_races = split_frames["validation"]["race_id"].drop_duplicates().sample(n=200, random_state=42)
    sampled_test_races = split_frames["test"]["race_id"].drop_duplicates().sample(n=200, random_state=42)
    train_rule_frame = split_frames["train"][split_frames["train"]["race_id"].isin(sampled_train_races)].copy()
    validation_eval_frame = split_frames["validation"][split_frames["validation"]["race_id"].isin(sampled_validation_races)].copy()
    test_eval_frame = split_frames["test"][split_frames["test"]["race_id"].isin(sampled_test_races)].copy()

    candidates = []
    for close_gap in (0.15, 0.35, 0.75):
        rules = learn_rules(train_rule_frame, close_gap=close_gap)
        validation_metrics = evaluate(validation_eval_frame, close_gap, rules)
        candidates.append(
            {
                "close_gap": close_gap,
                "rule_count": len(rules),
                "validation_metrics": validation_metrics,
                "selection_score": 0.5 * validation_metrics["exact_order_accuracy"]
                + 0.3 * validation_metrics["top3_set_accuracy"]
                + 0.2 * validation_metrics["top5_set_accuracy"],
                "rules": rules,
            }
        )

    best = max(
        candidates,
        key=lambda item: (
            item["selection_score"],
            item["validation_metrics"]["exact_order_accuracy"],
            item["validation_metrics"]["top3_set_accuracy"],
            item["validation_metrics"]["mean_kendall_tau"],
        ),
    )

    test_metrics = evaluate(test_eval_frame, best["close_gap"], best["rules"])
    result = {
        "best_close_gap": best["close_gap"],
        "rule_count": best["rule_count"],
        "validation_metrics": best["validation_metrics"],
        "test_metrics": test_metrics,
        "rules": best["rules"],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RULES_PATH.write_text(json.dumps(result, indent=2))
    METRICS_PATH.write_text(json.dumps({
        "validation": best["validation_metrics"],
        "test": test_metrics,
    }, indent=2))

    print("Candidates:")
    for item in candidates:
        print(
            json.dumps(
                {
                    "close_gap": item["close_gap"],
                    "rule_count": item["rule_count"],
                    "selection_score": round(item["selection_score"], 6),
                    "validation": item["validation_metrics"],
                }
            )
        )
    print("\nBest:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
