import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make the project root importable so tests can `import used_car_price_prediction`.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _synth(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    brands = ["Toyota", "Honda", "Ford", "BMW", "Tesla"]
    models = ["A", "B", "C", "D", "E", "F"]
    fuels = ["Gasoline", "Hybrid", "Electric", "Diesel"]
    trans = ["Automatic", "Manual"]

    model_year = rng.integers(2000, 2024, size=n)
    age = 2025 - model_year
    milage = rng.integers(5_000, 200_000, size=n)
    hp = rng.integers(120, 450, size=n)
    brand = rng.choice(brands, size=n)
    model = rng.choice(models, size=n)
    fuel = rng.choice(fuels, size=n)
    tr = rng.choice(trans, size=n)
    accident = rng.choice(["No", "Yes"], size=n, p=[0.8, 0.2])
    clean = rng.choice(["Yes", "No"], size=n, p=[0.85, 0.15])

    base = 35_000 - 1500 * age - 0.08 * milage + 60 * hp
    base += np.where(brand == "BMW", 8_000, 0)
    base += np.where(brand == "Tesla", 12_000, 0)
    base += np.where(accident == "Yes", -3_000, 0)
    base += np.where(clean == "No", -2_000, 0)
    noise = rng.normal(0, 2_000, size=n)
    price = np.clip(base + noise, 1_500, None)

    return pd.DataFrame(
        {
            "price": price,
            "model_year": model_year,
            "milage": milage,
            "brand": brand,
            "model": model,
            "engine": [f"{h}HP 2.5L 4 Cylinder" for h in hp],
            "fuel_type": fuel,
            "transmission": tr,
            "accident": accident,
            "clean_title": clean,
        }
    )


@pytest.fixture
def synth_df():
    return _synth(n=600, seed=0)


@pytest.fixture
def tiny_df():
    return _synth(n=60, seed=1)
