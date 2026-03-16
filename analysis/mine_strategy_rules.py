#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
REVERSE_DIR = ROOT / "analysis" / "reverse_engineering"
MATCHUP_PATH = REVERSE_DIR / "common_sequence_matchups.csv"
ONE_STOP_PATH = REVERSE_DIR / "one_stop_fraction_summary.csv"
OUTPUT_DIR = ROOT / "analysis" / "rule_mining"
RULES_PATH = OUTPUT_DIR / "strategy_rules.json"
SEQUENCE_RULES_PATH = OUTPUT_DIR / "sequence_dominance_rules.csv"
STOP_RULES_PATH = OUTPUT_DIR / "one_stop_timing_rules.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


def mine_sequence_rules(matchups: pd.DataFrame) -> pd.DataFrame:
    rules = matchups.copy()
    rules["edge"] = (rules["left_win_rate"] - 0.5).abs()
    rules["winner_sequence"] = rules.apply(
        lambda row: row["left_sequence"] if row["left_win_rate"] >= 0.5 else row["right_sequence"],
        axis=1,
    )
    rules["loser_sequence"] = rules.apply(
        lambda row: row["right_sequence"] if row["left_win_rate"] >= 0.5 else row["left_sequence"],
        axis=1,
    )
    rules["winner_win_rate"] = rules["left_win_rate"].where(
        rules["left_win_rate"] >= 0.5,
        1.0 - rules["left_win_rate"],
    )
    rules["mean_rank_gap_for_winner"] = rules["mean_left_rank_gap"].where(
        rules["left_win_rate"] >= 0.5,
        -rules["mean_left_rank_gap"],
    )
    rules["winner_first_stop_frac"] = rules["mean_left_first_stop_frac"].where(
        rules["left_win_rate"] >= 0.5,
        rules["mean_right_first_stop_frac"],
    )
    rules["loser_first_stop_frac"] = rules["mean_right_first_stop_frac"].where(
        rules["left_win_rate"] >= 0.5,
        rules["mean_left_first_stop_frac"],
    )
    rules = rules[
        [
            "temp_band",
            "lap_band",
            "winner_sequence",
            "loser_sequence",
            "pairings",
            "winner_win_rate",
            "mean_rank_gap_for_winner",
            "winner_first_stop_frac",
            "loser_first_stop_frac",
            "edge",
        ]
    ]
    rules = rules[(rules["pairings"] >= 200) & (rules["winner_win_rate"] >= 0.7)].copy()
    rules = rules.sort_values(
        ["edge", "pairings", "mean_rank_gap_for_winner"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return rules.round(4)


def mine_one_stop_rules(one_stop: pd.DataFrame) -> pd.DataFrame:
    rules = one_stop.copy()
    rules = rules[rules["rows"] >= 150].copy()
    rules["quality_score"] = (
        0.50 * (1.0 - (rules["mean_finish_rank"] - 1.0) / 19.0)
        + 0.25 * rules["podium_rate"]
        + 0.25 * rules["top5_rate"]
    )
    winners = (
        rules.sort_values(
            ["tire_sequence", "temp_band", "lap_band", "quality_score", "rows"],
            ascending=[True, True, True, False, False],
        )
        .groupby(["tire_sequence", "temp_band", "lap_band"], observed=True)
        .head(1)
        .copy()
    )
    winners = winners[
        [
            "tire_sequence",
            "temp_band",
            "lap_band",
            "first_stop_frac_bin",
            "rows",
            "mean_finish_rank",
            "podium_rate",
            "top5_rate",
            "quality_score",
        ]
    ]
    winners = winners.sort_values(
        ["quality_score", "rows", "mean_finish_rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return winners.round(4)


def build_rule_summary(sequence_rules: pd.DataFrame, stop_rules: pd.DataFrame) -> dict:
    return {
        "strongest_sequence_dominance_rules": sequence_rules.head(20).to_dict(orient="records"),
        "best_one_stop_timing_rules": stop_rules.head(30).to_dict(orient="records"),
        "lap_band_winners": (
            sequence_rules.groupby("lap_band", observed=True)
            .head(8)
            .to_dict(orient="records")
        ),
        "temp_band_winners": (
            sequence_rules.groupby("temp_band", observed=True)
            .head(8)
            .to_dict(orient="records")
        ),
    }


def main():
    matchups = pd.read_csv(MATCHUP_PATH)
    one_stop = pd.read_csv(ONE_STOP_PATH)

    sequence_rules = mine_sequence_rules(matchups)
    stop_rules = mine_one_stop_rules(one_stop)
    summary = build_rule_summary(sequence_rules, stop_rules)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sequence_rules.to_csv(SEQUENCE_RULES_PATH, index=False)
    stop_rules.to_csv(STOP_RULES_PATH, index=False)
    RULES_PATH.write_text(json.dumps(summary, indent=2))
    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "sequence_rule_count": int(len(sequence_rules)),
                "one_stop_rule_count": int(len(stop_rules)),
                "top_sequence_rules": summary["strongest_sequence_dominance_rules"][:10],
                "top_stop_rules": summary["best_one_stop_timing_rules"][:10],
            },
            indent=2,
        )
    )

    print("Top sequence dominance rules:")
    print(sequence_rules.head(20).to_string(index=False))
    print("\nTop one-stop timing rules:")
    print(stop_rules.head(20).to_string(index=False))
    print(f"\nWrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
