"""
Used Car Price Prediction — Stacked Ensemble

A regression pipeline that predicts used car prices. The script is intentionally
non-interactive and leakage-free:

  - Target encoding is fit inside an sklearn transformer that sees ONLY the
    training fold (no peeking at the held-out test set).
  - The data is split into train / test before any encoding or imputation that
    depends on the target. The test set's R² is the headline metric.
  - The script accepts a CSV path on the command line (no input() prompts) so
    it can run in CI and in scripts.

Run:
    python used_car_price_prediction.py --data path/to/cars.csv

See README.md for the expected CSV schema and a worked example.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

import lightgbm as lgb

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


CURRENT_YEAR = 2025
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def extract_horsepower(engine_str) -> float:
    """Pull the leading number from an engine string like '300HP 3.5L V6'."""
    if pd.isna(engine_str):
        return np.nan
    s = str(engine_str)
    if "HP" not in s:
        return np.nan
    try:
        return float(s.split("HP")[0].strip())
    except (ValueError, IndexError):
        return np.nan


def extract_base_model(model_str) -> str:
    if pd.isna(model_str):
        return "Unknown"
    parts = str(model_str).split()
    if len(parts) == 0:
        return "Unknown"
    if len(parts) == 1:
        return parts[0]
    return parts[0] + " " + parts[1]


def engineer_features(df: pd.DataFrame, current_year: int = CURRENT_YEAR) -> pd.DataFrame:
    """Apply deterministic feature engineering. No target-aware logic here."""
    df = df.copy()

    if "engine" in df.columns:
        df["horsepower"] = df["engine"].apply(extract_horsepower)
        df["horsepower_squared"] = df["horsepower"] ** 2
        df["horsepower_log"] = np.log1p(df["horsepower"])

    if "model_year" in df.columns:
        df["car_age"] = current_year - df["model_year"]
        df["car_age_squared"] = df["car_age"] ** 2
        df["car_age_log"] = np.log1p(df["car_age"].clip(lower=0))
        df["depreciation_factor"] = np.exp(-0.15 * df["car_age"])

    if "accident" in df.columns:
        df["had_accident"] = df["accident"].fillna("No").apply(
            lambda x: 0 if str(x).lower() in {"no", "0", "none reported"} else 1
        )

    if "clean_title" in df.columns:
        df["is_clean_title"] = df["clean_title"].fillna("No").apply(
            lambda x: 1 if str(x).lower() == "yes" else 0
        )

    if "milage" in df.columns and "car_age" in df.columns:
        df["log_milage"] = np.log1p(df["milage"])
        age_safe = df["car_age"].replace(0, 0.5)
        df["miles_per_year"] = df["milage"] / age_safe
        df["log_miles_per_year"] = np.log1p(df["miles_per_year"].clip(lower=0))
        expected = df["car_age"].clip(lower=1) * 12000
        df["milage_vs_expected"] = df["milage"] / expected
        df["high_milage_for_age"] = (df["milage_vs_expected"] > 1.2).astype(int)
        df["low_milage_for_age"] = (df["milage_vs_expected"] < 0.8).astype(int)

    if "fuel_type" in df.columns:
        for fuel in ["electric", "hybrid", "diesel", "gas"]:
            df[f"is_{fuel}"] = (
                df["fuel_type"].fillna("").str.lower().str.contains(fuel, na=False).astype(int)
            )

    if "model" in df.columns:
        df["model_base"] = df["model"].apply(extract_base_model)

    if "transmission" in df.columns:
        df["is_automatic"] = (
            df["transmission"].fillna("").str.lower().str.contains("auto", na=False).astype(int)
        )

    return df


# ---------------------------------------------------------------------------
# Target encoder — fit only on training data
# ---------------------------------------------------------------------------


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Bayesian-smoothed target encoder. Fits ONLY on training data.

    Parameters
    ----------
    cols : list of str
        Columns to encode in-place.
    smoothing : float
        Bayesian smoothing strength. Larger -> stronger pull toward global mean.
    min_samples : int
        Categories with fewer rows than this collapse to the global mean.
    """

    def __init__(self, cols: list[str], smoothing: float = 10.0, min_samples: int = 5):
        self.cols = cols
        self.smoothing = smoothing
        self.min_samples = min_samples

    def fit(self, X: pd.DataFrame, y):
        if y is None:
            raise ValueError("TargetEncoder requires y at fit time.")
        y = pd.Series(np.asarray(y), index=X.index)
        self.global_mean_ = float(y.mean())
        self.encoders_ = {}
        for col in self.cols:
            if col not in X.columns:
                continue
            cats = X[col].fillna("__MISSING__")
            tmp = pd.DataFrame({"cat": cats, "y": y.values})
            agg = tmp.groupby("cat")["y"].agg(["mean", "count"])
            mapping = {}
            for cat, (mean, count) in agg.iterrows():
                if count >= self.min_samples:
                    w = count / (count + self.smoothing)
                    mapping[cat] = float(w * mean + (1 - w) * self.global_mean_)
                else:
                    mapping[cat] = self.global_mean_
            self.encoders_[col] = mapping
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, mapping in self.encoders_.items():
            if col not in X.columns:
                continue
            X[col] = X[col].fillna("__MISSING__").map(mapping).fillna(self.global_mean_).astype(float)
        return X


# ---------------------------------------------------------------------------
# Data loading + train/test split
# ---------------------------------------------------------------------------


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_cols: list[str]
    cat_cols: list[str]
    log_transform: bool


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    log_transform: bool = True,
    random_state: int = RANDOM_STATE,
) -> PreparedData:
    """Engineer features, IQR-clip price, then SPLIT before any target-aware step.

    Returns a PreparedData with train/test slices ready for fitting the encoder.
    """
    if "price" not in df.columns:
        raise ValueError("DataFrame must contain a 'price' column.")

    df = engineer_features(df)

    # Drop obvious garbage rows first (cheap, target-free).
    df = df[df["price"] > 100].copy()

    drop_cols = [c for c in ["price", "id", "engine", "ext_col", "int_col", "accident", "clean_title"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y_price = df["price"].astype(float)

    X_train, X_test, y_train_price, y_test_price = train_test_split(
        X, y_price, test_size=test_size, random_state=random_state
    )

    # IQR upper bound computed on TRAIN only, applied to both halves so the
    # held-out set never informs the cutoff.
    Q1 = y_train_price.quantile(0.25)
    Q3 = y_train_price.quantile(0.75)
    upper = Q3 + 1.5 * (Q3 - Q1)
    train_mask = y_train_price <= upper
    test_mask = y_test_price <= upper
    X_train, y_train_price = X_train[train_mask], y_train_price[train_mask]
    X_test, y_test_price = X_test[test_mask], y_test_price[test_mask]

    # Median imputation for numeric cols, again fit on TRAIN, applied to both.
    numeric_cols = [c for c in X_train.columns if not pd.api.types.is_object_dtype(X_train[c]) and not pd.api.types.is_string_dtype(X_train[c])]
    medians = X_train[numeric_cols].median()
    X_train[numeric_cols] = X_train[numeric_cols].fillna(medians)
    X_test[numeric_cols] = X_test[numeric_cols].fillna(medians)

    cat_cols = [c for c in X_train.columns if pd.api.types.is_object_dtype(X_train[c]) or pd.api.types.is_string_dtype(X_train[c])]

    if log_transform:
        y_train = np.log1p(y_train_price)
        y_test = np.log1p(y_test_price)
    else:
        y_train = y_train_price
        y_test = y_test_price

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_cols=list(X.columns),
        cat_cols=cat_cols,
        log_transform=log_transform,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_stacked_ensemble(random_state: int = RANDOM_STATE) -> StackingRegressor:
    base = [
        (
            "lgb",
            lgb.LGBMRegressor(
                n_estimators=2000,
                max_depth=12,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                min_child_samples=20,
                verbose=-1,
                n_jobs=-1,
                random_state=random_state,
            ),
        )
    ]
    if XGBOOST_AVAILABLE:
        base.append(
            (
                "xgb",
                xgb.XGBRegressor(
                    n_estimators=2000,
                    max_depth=10,
                    learning_rate=0.02,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    gamma=0.1,
                    verbosity=0,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            )
        )
    if CATBOOST_AVAILABLE:
        base.append(
            (
                "cat",
                CatBoostRegressor(
                    iterations=2000,
                    depth=10,
                    learning_rate=0.02,
                    l2_leaf_reg=3,
                    random_strength=0.1,
                    verbose=0,
                    thread_count=-1,
                    random_seed=random_state,
                ),
            )
        )

    meta = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        verbose=-1,
        n_jobs=-1,
        random_state=random_state,
    )
    return StackingRegressor(estimators=base, final_estimator=meta, cv=5, n_jobs=-1)


def evaluate(model, X_test, y_test, log_transform: bool) -> dict:
    """Report metrics on the original price scale (not log)."""
    y_pred = model.predict(X_test)
    if log_transform:
        y_true_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred)
    else:
        y_true_orig = y_test
        y_pred_orig = y_pred

    return {
        "r2_log": r2_score(y_test, y_pred),
        "r2_price": r2_score(y_true_orig, y_pred_orig),
        "rmse_price": float(np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))),
        "mae_price": float(mean_absolute_error(y_true_orig, y_pred_orig)),
    }


# ---------------------------------------------------------------------------
# End-to-end pipeline + persistence
# ---------------------------------------------------------------------------


class CarPriceModel:
    """Inference wrapper. Carries the fitted encoder + estimator + log flag."""

    def __init__(self, encoder: TargetEncoder, model, log_transform: bool, feature_cols: list[str]):
        self.encoder = encoder
        self.model = model
        self.log_transform = log_transform
        self.feature_cols = feature_cols

    def predict(self, X_raw: pd.DataFrame) -> np.ndarray:
        X = engineer_features(X_raw)
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.feature_cols]
        X = self.encoder.transform(X)
        for col in X.columns:
            if X[col].dtype != "object" and X[col].isna().any():
                X[col] = X[col].fillna(0)
        preds = self.model.predict(X)
        return np.expm1(preds) if self.log_transform else preds


def run_pipeline(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    log_transform: bool = True,
    random_state: int = RANDOM_STATE,
) -> tuple[CarPriceModel, dict]:
    """Fit the ensemble on train, evaluate on held-out test, return wrapper."""
    prepared = prepare_data(df, test_size=test_size, log_transform=log_transform, random_state=random_state)

    encoder = TargetEncoder(cols=prepared.cat_cols).fit(prepared.X_train, prepared.y_train)
    X_train_enc = encoder.transform(prepared.X_train)
    X_test_enc = encoder.transform(prepared.X_test)

    model = build_stacked_ensemble(random_state=random_state)
    model.fit(X_train_enc, prepared.y_train)

    metrics = evaluate(model, X_test_enc, prepared.y_test, prepared.log_transform)

    wrapper = CarPriceModel(
        encoder=encoder,
        model=model,
        log_transform=prepared.log_transform,
        feature_cols=list(prepared.X_train.columns),
    )
    return wrapper, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Used car price prediction pipeline.")
    parser.add_argument("--data", required=True, help="Path to a CSV with a 'price' column.")
    parser.add_argument("--output", default="car_price_model.joblib", help="Where to save the fitted model.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--no-log-transform", action="store_true", help="Disable log1p on the price target.")
    args = parser.parse_args(argv)

    if not os.path.exists(args.data):
        print(f"ERROR: data file not found: {args.data}")
        return 2

    t0 = time.time()
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")

    wrapper, metrics = run_pipeline(
        df,
        test_size=args.test_size,
        log_transform=not args.no_log_transform,
        random_state=args.seed,
    )

    print("\nHeld-out test metrics (price scale, USD):")
    print(f"  R²   : {metrics['r2_price']:.4f}")
    print(f"  RMSE : ${metrics['rmse_price']:,.0f}")
    print(f"  MAE  : ${metrics['mae_price']:,.0f}")
    print(f"  R² (log target): {metrics['r2_log']:.4f}")

    joblib.dump(wrapper, args.output)
    print(f"\nSaved model to {args.output}")
    print(f"Total time: {(time.time() - t0):.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
