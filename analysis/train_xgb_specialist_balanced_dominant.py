#!/usr/bin/env python3
import os

import train_xgb_ranker as base


MODEL_TAG = os.environ.get("MODEL_TAG", "balanced_dominant")


def race_filter(frame):
    race_level = (
        frame.groupby("race_id", sort=False)
        .agg(
            total_laps=("total_laps", "first"),
            track_temp=("track_temp", "first"),
            unique_sequence_count=("tire_sequence", "nunique"),
            two_stop_drivers=("pit_stop_count", lambda s: int((s == 2).sum())),
            top_sequence=("tire_sequence", lambda s: str(s.value_counts().index[0])),
            top_sequence_count=("tire_sequence", lambda s: int(s.value_counts().iloc[0])),
        )
        .reset_index()
    )
    mask = (
        (race_level["two_stop_drivers"] == 0)
        & (race_level["total_laps"] <= 45)
        & (race_level["track_temp"] >= 28)
        & (race_level["track_temp"] <= 36)
        & (race_level["unique_sequence_count"] >= 4)
        & (race_level["unique_sequence_count"] <= 6)
        & (race_level["top_sequence_count"] >= 6)
        & race_level["top_sequence"].isin(["MEDIUM>HARD", "SOFT>HARD"])
    )
    return set(race_level.loc[mask, "race_id"])


def main():
    df, splits, v2_record, v4_record = base.load_inputs()
    df = base.add_physics_predictions(df, v2_record, v4_record)
    df = base.add_regime_features(df)

    specialist_races = race_filter(df)
    train = df[df["race_id"].isin(splits["train"]) & df["race_id"].isin(specialist_races)].copy()
    validation = df[df["race_id"].isin(splits["validation"]) & df["race_id"].isin(specialist_races)].copy()
    test = df[df["race_id"].isin(splits["test"]) & df["race_id"].isin(specialist_races)].copy()

    if train["race_id"].nunique() == 0:
        raise SystemExit("No dominant balanced specialist training races matched the filter")

    major_sequences = base.derive_major_sequences(train)
    full_df = base.add_race_composition_features(df, major_sequences)
    train = full_df[full_df["race_id"].isin(train["race_id"].unique())].copy()
    validation = full_df[full_df["race_id"].isin(validation["race_id"].unique())].copy()
    test = full_df[full_df["race_id"].isin(test["race_id"].unique())].copy()
    feature_columns, categorical_columns = base.get_feature_config(major_sequences)

    available_races = train["race_id"].drop_duplicates()
    sampled_train_races = available_races.sample(
        n=min(base.TRAIN_RACES, len(available_races)),
        random_state=42,
    )
    train = train[train["race_id"].isin(sampled_train_races)].copy()

    train_x, validation_x, test_x, categories, encoded_columns = base.one_hot_encode(
        train[feature_columns],
        validation[feature_columns],
        test[feature_columns],
        categorical_columns,
    )

    ranker = base.XGBRanker(
        objective="rank:pairwise",
        tree_method="hist",
        learning_rate=base.LEARNING_RATE,
        n_estimators=base.N_ESTIMATORS,
        max_depth=base.MAX_DEPTH,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.5,
        n_jobs=1,
        random_state=42,
    )

    ranker.fit(
        train_x,
        train["finish_rank"],
        group=base.group_sizes(train),
        eval_set=[(validation_x, validation["finish_rank"])],
        eval_group=[base.group_sizes(validation)],
        verbose=False,
    )

    validation_metrics = base.evaluate(validation, ranker.predict(validation_x))
    test_metrics = base.evaluate(test, ranker.predict(test_x))

    model_dir = base.ROOT / "analysis" / "models" / f"xgb_ranker_{MODEL_TAG}"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metrics.json").write_text(
        base.json.dumps(
            {
                "validation": {"xgb_ranker": validation_metrics},
                "test": {"xgb_ranker": test_metrics},
                "specialist_filter": {
                    "train_races": int(train["race_id"].nunique()),
                    "validation_races": int(validation["race_id"].nunique()),
                    "test_races": int(test["race_id"].nunique()),
                },
            },
            indent=2,
        )
    )
    ranker.save_model(str(model_dir / "model.json"))
    (model_dir / "metadata.json").write_text(
        base.json.dumps(
            {
                "feature_columns": feature_columns,
                "categorical_columns": categorical_columns,
                "categories": categories,
                "encoded_columns": encoded_columns,
                "major_sequences": major_sequences,
                "use_composition_features": True,
                "model_tag": MODEL_TAG,
            },
            indent=2,
        )
    )
    print(
        base.json.dumps(
            {
                "validation": {"xgb_ranker": validation_metrics},
                "test": {"xgb_ranker": test_metrics},
                "specialist_filter": {
                    "train_races": int(train["race_id"].nunique()),
                    "validation_races": int(validation["race_id"].nunique()),
                    "test_races": int(test["race_id"].nunique()),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
