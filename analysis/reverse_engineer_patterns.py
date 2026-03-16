#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "analysis" / "historical_driver_features.csv"
OUTPUT_DIR = ROOT / "analysis" / "reverse_engineering"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
SEQUENCE_SUMMARY_PATH = OUTPUT_DIR / "sequence_summary.csv"
SEQUENCE_TEMP_PATH = OUTPUT_DIR / "sequence_by_temp_band.csv"
SEQUENCE_LAP_PATH = OUTPUT_DIR / "sequence_by_lap_band.csv"
ONE_STOP_FRACTION_PATH = OUTPUT_DIR / "one_stop_fraction_summary.csv"
MATCHUP_PATH = OUTPUT_DIR / "common_sequence_matchups.csv"


COMMON_MATCHUPS = [
    ("SOFT>HARD", "MEDIUM>HARD"),
    ("HARD>MEDIUM", "MEDIUM>HARD"),
    ("SOFT>MEDIUM", "MEDIUM>SOFT"),
    ("SOFT>HARD>MEDIUM", "MEDIUM>HARD>SOFT"),
    ("SOFT>MEDIUM>HARD", "MEDIUM>SOFT>HARD"),
    ("HARD>SOFT", "HARD>MEDIUM"),
]


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["temp_band"] = pd.cut(
        out["track_temp"],
        bins=[17, 24, 30, 36, 42],
        labels=["cool", "mild", "warm", "hot"],
    )
    out["lap_band"] = pd.cut(
        out["total_laps"],
        bins=[24, 35, 45, 55, 70],
        labels=["short", "mid_short", "mid", "long"],
    )
    out["first_stop_frac"] = np.where(
        out["first_stop_lap"].fillna(0) > 0,
        out["first_stop_lap"].fillna(0) / out["total_laps"],
        np.nan,
    )
    out["second_stop_frac"] = np.where(
        out["second_stop_lap"].fillna(0) > 0,
        out["second_stop_lap"].fillna(0) / out["total_laps"],
        np.nan,
    )
    for idx in (1, 2, 3):
        out[f"stint_{idx}_frac"] = out[f"stint_{idx}_laps"] / out["total_laps"]
    out["soft_frac"] = out["soft_laps"] / out["total_laps"]
    out["medium_frac"] = out["medium_laps"] / out["total_laps"]
    out["hard_frac"] = out["hard_laps"] / out["total_laps"]
    out["winner_flag"] = (out["finish_rank"] == 1).astype(int)
    out["podium_flag"] = (out["finish_rank"] <= 3).astype(int)
    out["top5_flag"] = (out["finish_rank"] <= 5).astype(int)
    return out


def sequence_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("tire_sequence")
        .agg(
            rows=("driver_id", "size"),
            mean_finish_rank=("finish_rank", "mean"),
            median_finish_rank=("finish_rank", "median"),
            win_rate=("winner_flag", "mean"),
            podium_rate=("podium_flag", "mean"),
            top5_rate=("top5_flag", "mean"),
            mean_total_laps=("total_laps", "mean"),
            mean_track_temp=("track_temp", "mean"),
            mean_first_stop_frac=("first_stop_frac", "mean"),
            mean_stint_1_frac=("stint_1_frac", "mean"),
            mean_final_stint_frac=("stint_3_frac", "mean"),
        )
        .sort_values(["rows", "mean_finish_rank"], ascending=[False, True])
        .reset_index()
    )
    return grouped.round(4)


def banded_sequence_summary(df: pd.DataFrame, band_col: str, top_sequences: list[str]) -> pd.DataFrame:
    subset = df[df["tire_sequence"].isin(top_sequences)].copy()
    summary = (
        subset.groupby([band_col, "tire_sequence"], observed=True)
        .agg(
            rows=("driver_id", "size"),
            mean_finish_rank=("finish_rank", "mean"),
            win_rate=("winner_flag", "mean"),
            podium_rate=("podium_flag", "mean"),
            top5_rate=("top5_flag", "mean"),
            mean_first_stop_frac=("first_stop_frac", "mean"),
            mean_stint_1_frac=("stint_1_frac", "mean"),
            mean_final_stint_frac=("stint_3_frac", "mean"),
        )
        .reset_index()
        .sort_values([band_col, "rows", "mean_finish_rank"], ascending=[True, False, True])
    )
    return summary.round(4)


def one_stop_fraction_summary(df: pd.DataFrame) -> pd.DataFrame:
    one_stop = df[df["pit_stop_count"] == 1].copy()
    common_sequences = one_stop["tire_sequence"].value_counts().head(6).index.tolist()
    one_stop = one_stop[one_stop["tire_sequence"].isin(common_sequences)].copy()
    one_stop["first_stop_frac_bin"] = pd.cut(
        one_stop["first_stop_frac"],
        bins=[0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
        labels=["0.0-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.7", "0.7-1.0"],
        include_lowest=True,
    )
    summary = (
        one_stop.groupby(["tire_sequence", "temp_band", "lap_band", "first_stop_frac_bin"], observed=True)
        .agg(
            rows=("driver_id", "size"),
            mean_finish_rank=("finish_rank", "mean"),
            podium_rate=("podium_flag", "mean"),
            top5_rate=("top5_flag", "mean"),
        )
        .reset_index()
        .sort_values(["tire_sequence", "temp_band", "lap_band", "first_stop_frac_bin"])
    )
    return summary.round(4)


def common_sequence_matchups(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left_seq, right_seq in COMMON_MATCHUPS:
        pair_df = df[df["tire_sequence"].isin([left_seq, right_seq])].copy()
        if pair_df.empty:
            continue

        for race_id, race in pair_df.groupby("race_id", sort=False):
            left = race[race["tire_sequence"] == left_seq]
            right = race[race["tire_sequence"] == right_seq]
            if left.empty or right.empty:
                continue
            race_meta = race.iloc[0]
            left_ranks = left["finish_rank"].to_numpy()
            right_ranks = right["finish_rank"].to_numpy()
            gap_matrix = right_ranks[None, :] - left_ranks[:, None]
            left_stop = left["first_stop_frac"].to_numpy()
            right_stop = right["first_stop_frac"].to_numpy()

            rows.append(
                {
                    "left_sequence": left_seq,
                    "right_sequence": right_seq,
                    "race_id": race_id,
                    "track": race_meta["track"],
                    "temp_band": race_meta["temp_band"],
                    "lap_band": race_meta["lap_band"],
                    "pairings": int(gap_matrix.size),
                    "left_win_count": int((gap_matrix > 0).sum()),
                    "sum_left_rank_gap": float(gap_matrix.sum()),
                    "sum_left_first_stop_frac": float(np.nansum(left_stop) * len(right_stop)),
                    "sum_right_first_stop_frac": float(np.nansum(right_stop) * len(left_stop)),
                }
            )

    matchup_df = pd.DataFrame(rows)
    if matchup_df.empty:
        return matchup_df

    summary = (
        matchup_df.groupby(["left_sequence", "right_sequence", "temp_band", "lap_band"], observed=True)
        .agg(
            pairings=("pairings", "sum"),
            left_win_count=("left_win_count", "sum"),
            sum_left_rank_gap=("sum_left_rank_gap", "sum"),
            sum_left_first_stop_frac=("sum_left_first_stop_frac", "sum"),
            sum_right_first_stop_frac=("sum_right_first_stop_frac", "sum"),
        )
        .assign(
            left_win_rate=lambda x: x["left_win_count"] / x["pairings"],
            mean_left_rank_gap=lambda x: x["sum_left_rank_gap"] / x["pairings"],
            mean_left_first_stop_frac=lambda x: x["sum_left_first_stop_frac"] / x["pairings"],
            mean_right_first_stop_frac=lambda x: x["sum_right_first_stop_frac"] / x["pairings"],
        )
        .drop(columns=["left_win_count", "sum_left_rank_gap", "sum_left_first_stop_frac", "sum_right_first_stop_frac"])
        .reset_index()
        .sort_values(["left_sequence", "right_sequence", "pairings"], ascending=[True, True, False])
    )
    return summary.round(4)


def build_summary_json(
    seq_df: pd.DataFrame,
    seq_temp_df: pd.DataFrame,
    seq_lap_df: pd.DataFrame,
    matchup_df: pd.DataFrame,
) -> dict:
    top_common = seq_df.head(8)["tire_sequence"].tolist()
    summary = {
        "top_common_sequences": seq_df.head(8).to_dict(orient="records"),
        "best_sequences_by_win_rate_min_1000_rows": (
            seq_df[seq_df["rows"] >= 1000]
            .sort_values(["win_rate", "podium_rate"], ascending=[False, False])
            .head(8)
            .to_dict(orient="records")
        ),
        "best_sequences_by_mean_finish_min_1000_rows": (
            seq_df[seq_df["rows"] >= 1000]
            .sort_values(["mean_finish_rank", "podium_rate"], ascending=[True, False])
            .head(8)
            .to_dict(orient="records")
        ),
        "sequence_temp_band_examples": (
            seq_temp_df[seq_temp_df["tire_sequence"].isin(top_common)]
            .groupby("temp_band", observed=True)
            .head(6)
            .to_dict(orient="records")
        ),
        "sequence_lap_band_examples": (
            seq_lap_df[seq_lap_df["tire_sequence"].isin(top_common)]
            .groupby("lap_band", observed=True)
            .head(6)
            .to_dict(orient="records")
        ),
        "strongest_matchups_min_200_pairings": (
            matchup_df[matchup_df["pairings"] >= 200]
            .assign(edge=lambda x: (x["left_win_rate"] - 0.5).abs())
            .sort_values(["edge", "pairings"], ascending=[False, False])
            .drop(columns=["edge"])
            .head(20)
            .to_dict(orient="records")
        ),
    }
    return summary


def main():
    df = pd.read_csv(DATA_PATH)
    df = add_derived_columns(df)

    seq_df = sequence_summary(df)
    top_sequences = seq_df.head(10)["tire_sequence"].tolist()
    seq_temp_df = banded_sequence_summary(df, "temp_band", top_sequences)
    seq_lap_df = banded_sequence_summary(df, "lap_band", top_sequences)
    one_stop_df = one_stop_fraction_summary(df)
    matchup_df = common_sequence_matchups(df)
    summary = build_summary_json(seq_df, seq_temp_df, seq_lap_df, matchup_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seq_df.to_csv(SEQUENCE_SUMMARY_PATH, index=False)
    seq_temp_df.to_csv(SEQUENCE_TEMP_PATH, index=False)
    seq_lap_df.to_csv(SEQUENCE_LAP_PATH, index=False)
    one_stop_df.to_csv(ONE_STOP_FRACTION_PATH, index=False)
    matchup_df.to_csv(MATCHUP_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print("Top common sequences:")
    print(seq_df.head(10).to_string(index=False))
    print("\nBest sequence/temperature slices:")
    print(
        seq_temp_df.sort_values(["temp_band", "mean_finish_rank", "rows"], ascending=[True, True, False])
        .groupby("temp_band", observed=True)
        .head(6)
        .to_string(index=False)
    )
    print("\nBest sequence/lap-band slices:")
    print(
        seq_lap_df.sort_values(["lap_band", "mean_finish_rank", "rows"], ascending=[True, True, False])
        .groupby("lap_band", observed=True)
        .head(6)
        .to_string(index=False)
    )
    print("\nMost directional common matchups:")
    if matchup_df.empty:
        print("No matchup rows")
    else:
        print(
            matchup_df[matchup_df["pairings"] >= 200]
            .assign(edge=lambda x: (x["left_win_rate"] - 0.5).abs())
            .sort_values(["edge", "pairings"], ascending=[False, False])
            .drop(columns=["edge"])
            .head(20)
            .to_string(index=False)
        )
    print(f"\nWrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
