import numpy as np
import pandas as pd
import pytest

from used_car_price_prediction import (
    engineer_features,
    extract_base_model,
    extract_horsepower,
)


def test_extract_horsepower_parses_typical_string():
    assert extract_horsepower("300HP 3.5L V6") == 300.0


def test_extract_horsepower_handles_missing_and_garbage():
    assert np.isnan(extract_horsepower(None))
    assert np.isnan(extract_horsepower("V8 5.0L"))
    assert np.isnan(extract_horsepower(float("nan")))


def test_extract_base_model_handles_short_and_long_names():
    assert extract_base_model("Camry") == "Camry"
    assert extract_base_model("Honda Civic LX") == "Honda Civic"
    assert extract_base_model(None) == "Unknown"


def test_engineer_features_creates_expected_columns(synth_df):
    out = engineer_features(synth_df)
    for col in [
        "horsepower",
        "car_age",
        "depreciation_factor",
        "had_accident",
        "is_clean_title",
        "log_milage",
        "miles_per_year",
        "model_base",
        "is_automatic",
    ]:
        assert col in out.columns, f"missing {col}"


def test_engineer_features_handles_zero_age_without_div_by_zero():
    df = pd.DataFrame(
        {
            "price": [20_000],
            "model_year": [2025],
            "milage": [10_000],
            "engine": ["200HP 2.0L"],
            "accident": ["No"],
            "clean_title": ["Yes"],
            "brand": ["Toyota"],
            "model": ["Camry"],
            "fuel_type": ["Gasoline"],
            "transmission": ["Automatic"],
        }
    )
    out = engineer_features(df)
    assert np.isfinite(out["miles_per_year"].iloc[0])
    assert np.isfinite(out["milage_vs_expected"].iloc[0])
