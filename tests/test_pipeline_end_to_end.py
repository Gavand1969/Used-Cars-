"""Smoke tests for the end-to-end pipeline on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from used_car_price_prediction import prepare_data, run_pipeline


def test_prepare_data_splits_disjoint_indices(synth_df):
    prepared = prepare_data(synth_df, test_size=0.25, random_state=0)
    train_idx = set(prepared.X_train.index.tolist())
    test_idx = set(prepared.X_test.index.tolist())
    assert train_idx.isdisjoint(test_idx)
    assert len(prepared.X_train) > 0 and len(prepared.X_test) > 0


def test_prepare_data_does_not_target_encode(synth_df):
    """After prepare_data the categorical columns are still strings/objects -
    encoding happens AFTER the split, inside run_pipeline."""
    prepared = prepare_data(synth_df, random_state=0)
    # 'brand' and 'model' must still be non-numeric (string/object) at split time
    assert not pd.api.types.is_numeric_dtype(prepared.X_train["brand"])
    assert not pd.api.types.is_numeric_dtype(prepared.X_train["model"])
    assert "brand" in prepared.cat_cols
    assert "model" in prepared.cat_cols


def test_run_pipeline_learns_something_on_synthetic_data(synth_df):
    wrapper, metrics = run_pipeline(synth_df, random_state=0)
    # On the synthetic dataset, a competent model should clear R²=0.5.
    assert metrics["r2_price"] > 0.5, f"Surprisingly low R²: {metrics}"
    assert metrics["rmse_price"] > 0
    assert metrics["mae_price"] > 0


def test_run_pipeline_wrapper_predicts_on_new_rows(synth_df):
    wrapper, _ = run_pipeline(synth_df, random_state=0)
    new = pd.DataFrame(
        [
            {
                "model_year": 2020,
                "milage": 35_000,
                "brand": "Toyota",
                "model": "A",
                "engine": "203HP 2.5L 4 Cylinder",
                "fuel_type": "Gasoline",
                "transmission": "Automatic",
                "accident": "No",
                "clean_title": "Yes",
            },
            {
                # Unseen brand should still get a price (falls back to global mean)
                "model_year": 2018,
                "milage": 80_000,
                "brand": "Bugatti",
                "model": "X",
                "engine": "1500HP V16",
                "fuel_type": "Gasoline",
                "transmission": "Automatic",
                "accident": "No",
                "clean_title": "Yes",
            },
        ]
    )
    preds = wrapper.predict(new)
    assert preds.shape == (2,)
    assert np.all(np.isfinite(preds))
    assert np.all(preds > 0)
