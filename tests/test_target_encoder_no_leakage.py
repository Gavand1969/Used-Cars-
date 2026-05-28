"""The critical anti-leakage test.

If the encoder ever starts peeking at the test set's target, this will fail.
"""

import numpy as np
import pandas as pd

from used_car_price_prediction import TargetEncoder


def test_encoder_uses_only_training_targets():
    train = pd.DataFrame({"brand": ["A", "A", "B", "B", "C"]})
    y_train = pd.Series([10.0, 12.0, 50.0, 52.0, 30.0])

    enc = TargetEncoder(cols=["brand"], smoothing=0.0, min_samples=1).fit(train, y_train)

    # New data introduces an unseen category 'Z'. It MUST fall back to the
    # training global mean rather than learning a value from y_test.
    test = pd.DataFrame({"brand": ["A", "Z"]})
    out = enc.transform(test)

    train_global_mean = float(y_train.mean())
    assert out.loc[1, "brand"] == train_global_mean, (
        "Unseen category should fall back to training global mean, not leak."
    )
    # Known category 'A' should be close to its training mean (11.0) with smoothing=0.
    assert abs(out.loc[0, "brand"] - 11.0) < 1e-9


def test_encoder_is_deterministic_under_fixed_inputs():
    df = pd.DataFrame({"brand": ["A", "B", "A", "C"]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])

    a = TargetEncoder(cols=["brand"]).fit(df, y).transform(df)
    b = TargetEncoder(cols=["brand"]).fit(df, y).transform(df)
    pd.testing.assert_frame_equal(a, b)


def test_encoder_does_not_mutate_input():
    df = pd.DataFrame({"brand": ["A", "B", "A"]})
    y = pd.Series([1.0, 2.0, 3.0])
    enc = TargetEncoder(cols=["brand"]).fit(df, y)
    original = df.copy()
    enc.transform(df)
    pd.testing.assert_frame_equal(df, original)


def test_encoder_handles_missing_values():
    df = pd.DataFrame({"brand": ["A", None, "A", "B"]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    enc = TargetEncoder(cols=["brand"], min_samples=1, smoothing=0.0).fit(df, y)
    out = enc.transform(df)
    # No NaNs after transform
    assert not out["brand"].isna().any()
