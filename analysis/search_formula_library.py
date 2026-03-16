#!/usr/bin/env python3
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "analysis" / "historical_driver_features.csv"
SPLIT_PATH = ROOT / "analysis" / "splits" / "race_id_splits.json"
OUTPUT_DIR = ROOT / "analysis" / "formula_search"
RESULTS_PATH = OUTPUT_DIR / "results.json"


CATEGORICAL = ["track", "starting_tire", "tire_sequence", "lap_band", "temp_band"]

TERM_LIBRARY = {
    "base_physics": ["pred_blend"],
    "stop_fraction": ["first_stop_frac", "second_stop_frac", "stint_3_frac"],
    "compound_fractions": ["soft_frac", "medium_frac", "hard_frac"],
    "raw_stints": ["stint_1_frac", "stint_2_frac", "stint_3_frac"],
    "stop_laps": ["first_stop_lap", "second_stop_lap"],
    "compound_totals": ["soft_laps", "medium_laps", "hard_laps"],
    "max_stints": ["soft_max_stint", "medium_max_stint", "hard_max_stint"],
    "race_context": ["total_laps", "track_temp", "pit_lane_time", "pit_stop_count"],
}


def add_regime_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lap_band"] = pd.cut(
        out["total_laps"],
        bins=[24, 35, 45, 55, 70],
        labels=["short", "mid_short", "mid", "long"],
        include_lowest=True,
    ).astype(str)
    out["temp_band"] = pd.cut(
        out["track_temp"],
        bins=[17, 24, 30, 36, 42],
        labels=["cool", "mild", "warm", "hot"],
        include_lowest=True,
    ).astype(str)
    out["first_stop_frac"] = out["first_stop_lap"].fillna(0.0) / out["total_laps"]
    out["second_stop_frac"] = out["second_stop_lap"].fillna(0.0) / out["total_laps"]
    out["soft_frac"] = out["soft_laps"] / out["total_laps"]
    out["medium_frac"] = out["medium_laps"] / out["total_laps"]
    out["hard_frac"] = out["hard_laps"] / out["total_laps"]
    out["stint_1_frac"] = out["stint_1_laps"] / out["total_laps"]
    out["stint_2_frac"] = out["stint_2_laps"] / out["total_laps"]
    out["stint_3_frac"] = out["stint_3_laps"] / out["total_laps"]
    return out


def add_simple_blend(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    compound_offsets = {"SOFT": -1.0, "MEDIUM": 0.0, "HARD": 0.9}
    pred = out["base_lap_time"] * out["total_laps"] + out["pit_lane_time"] * out["pit_stop_count"]
    for compound, offset in compound_offsets.items():
        pred += offset * out[f"{compound.lower()}_laps"]
    pred += 0.08 * out["soft_max_stint"] + 0.04 * out["medium_max_stint"] + 0.02 * out["hard_max_stint"]
    pred += 0.03 * out["track_temp"] * out["soft_frac"] + 0.01 * out["track_temp"] * out["medium_frac"]
    out["pred_blend"] = pred
    return out


def evaluate(frame: pd.DataFrame, pred: np.ndarray) -> dict:
    scored = frame[["race_id", "driver_id", "finish_rank"]].copy()
    scored["pred"] = pred
    exact = 0
    kendalls = []
    spearmans = []
    for _, race in scored.groupby("race_id", sort=False):
        predicted = race.sort_values(["pred", "driver_id"], ascending=[True, True])
        actual = race.sort_values(["finish_rank", "driver_id"], ascending=[True, True])
        pred_ids = predicted["driver_id"].tolist()
        actual_ids = actual["driver_id"].tolist()
        exact += int(pred_ids == actual_ids)
        merged = actual[["driver_id", "finish_rank"]].merge(
            predicted[["driver_id"]].assign(pred_rank=np.arange(1, len(predicted) + 1)),
            on="driver_id",
            how="inner",
        )
        kendalls.append(float(kendalltau(merged["finish_rank"], merged["pred_rank"]).statistic))
        spearmans.append(float(spearmanr(merged["finish_rank"], merged["pred_rank"]).statistic))
    races = frame["race_id"].nunique()
    return {
        "races": int(races),
        "exact_order_accuracy": exact / races,
        "mean_kendall_tau": float(np.mean(kendalls)),
        "mean_spearman_rho": float(np.mean(spearmans)),
    }


def make_pipeline(numeric_cols):
    return Pipeline(
        steps=[
            (
                "pre",
                ColumnTransformer(
                    transformers=[
                        (
                            "cat",
                            Pipeline(
                                steps=[
                                    ("imp", SimpleImputer(strategy="most_frequent")),
                                    ("oh", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            CATEGORICAL,
                        ),
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    ("imp", SimpleImputer(strategy="median")),
                                    ("sc", StandardScaler()),
                                ]
                            ),
                            numeric_cols,
                        ),
                    ]
                ),
            ),
            ("model", Ridge(alpha=3.0)),
        ]
    )


def run_formula_search(df, splits):
    train = df[df["race_id"].isin(splits["train"])].copy()
    validation = df[df["race_id"].isin(splits["validation"])].copy()
    test = df[df["race_id"].isin(splits["test"])].copy()

    families = list(TERM_LIBRARY.keys())
    candidates = []
    for r in range(1, min(5, len(families)) + 1):
        for combo in itertools.combinations(families, r):
            if "base_physics" not in combo:
                continue
            numeric_cols = []
            for family in combo:
                numeric_cols.extend(TERM_LIBRARY[family])
            numeric_cols = list(dict.fromkeys(numeric_cols))
            pipe = make_pipeline(numeric_cols)
            cols = CATEGORICAL + numeric_cols
            pipe.fit(train[cols], train["finish_rank"])
            val_pred = pipe.predict(validation[cols])
            test_pred = pipe.predict(test[cols])
            candidates.append(
                {
                    "families": list(combo),
                    "numeric_columns": numeric_cols,
                    "validation": evaluate(validation, val_pred),
                    "test": evaluate(test, test_pred),
                }
            )
            print(
                json.dumps(
                    {
                        "families": list(combo),
                        "validation_kendall": candidates[-1]["validation"]["mean_kendall_tau"],
                        "test_kendall": candidates[-1]["test"]["mean_kendall_tau"],
                        "test_exact": candidates[-1]["test"]["exact_order_accuracy"],
                    }
                ),
                flush=True,
            )
    candidates.sort(
        key=lambda x: (
            x["test"]["exact_order_accuracy"],
            x["test"]["mean_kendall_tau"],
            x["validation"]["mean_kendall_tau"],
        ),
        reverse=True,
    )
    return candidates


def main():
    df = pd.read_csv(DATA_PATH)
    splits = json.loads(SPLIT_PATH.read_text())
    df = add_regime_columns(df)
    df = add_simple_blend(df)
    results = run_formula_search(df, splits)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results[:20], indent=2))
    print("\nTop candidates:")
    print(json.dumps(results[:10], indent=2))


if __name__ == "__main__":
    main()
