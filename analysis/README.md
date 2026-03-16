# Analysis Workflow

This folder is the starting point for a proper modeling workflow.

## Principles

- Treat this as a ranking problem, not a standard multiclass classification problem.
- Do not encode `driver_id` or `grid_slot` as numeric signal. The problem statement says cars and drivers are identical.
- Keep categorical features categorical: `track`, `starting_tire`, `tire_sequence`, and per-stint compounds should be one-hot encoded or embedded, not ordinal-encoded blindly.
- Keep numerical features numerical: `total_laps`, `base_lap_time`, `pit_lane_time`, `track_temp`, stop laps, stint lengths, and compound lap totals.
- Do not resample by class. Every race already contains a full 1-20 ranking, so this is not an imbalanced label-frequency problem in the usual sense.
- Focus on leakage control. The bundled `data/test_cases/expected_outputs/` should never be used for training or feature engineering.

## EDA Goals

1. Confirm schema consistency and missingness.
2. Separate categorical and numerical fields before modeling.
3. Inspect outliers in temperature, pit windows, stint lengths, and stop counts.
4. Validate that training and test covariate ranges overlap.
5. Build engineered features at the race-driver level for later ranking models.

## Current Findings

- Historical dataset size: 30,000 races and 600,000 race-driver rows.
- Tracks are balanced across 7 categories.
- `pit_stop_count` is almost entirely 1-stop or 2-stop strategies.
- All records are structurally complete; no missing values were found in the supplied JSON.
- Test-set ranges for track temperature and total laps are inside the training ranges, so covariate shift is limited.

## Files

- [build_dataset.py](/Users/k.e.oshada/Documents/box-box-box/analysis/build_dataset.py): flattens race JSON into a modeling table.
- [eda.ipynb](/Users/k.e.oshada/Documents/box-box-box/analysis/eda.ipynb): notebook for profiling and visualization.
