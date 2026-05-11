# Used Car Price Prediction — Stacked Ensemble Model

A high-performance machine learning pipeline that predicts used car prices using a stacked ensemble of gradient boosting models. Built for accuracy with production-grade feature engineering and model persistence.

---

## Model Performance

| Metric | Value |
|--------|-------|
| R² Score | ~0.93+ |
| Algorithm | Stacked Ensemble |
| Base Models | LightGBM + XGBoost + CatBoost |
| Meta-learner | LightGBM |

---

## What Makes This Advanced

### Stacked Ensemble
Three gradient boosting models are trained independently, then their predictions are fed as features into a LightGBM meta-learner. This captures patterns that no single model can learn alone.

```
Input Features
     │
     ├──► LightGBM ──► predictions ─┐
     ├──► XGBoost  ──► predictions ─┼──► LightGBM Meta-learner ──► Final Price
     └──► CatBoost ──► predictions ─┘
```

### Feature Engineering
- **Horsepower extraction** from raw engine strings
- **Depreciation curve** via exponential decay: `e^(-0.15 × age)`
- **Mileage vs. expected** — flags over/under-mileage vehicles
- **Non-linear transforms** — log and squared terms for age, mileage, horsepower
- **Binary indicators** — accident history, clean title, fuel type, transmission

### Target Encoding
Categorical features (brand, model, transmission) are encoded using regularised target means with Bayesian smoothing — avoids the high cardinality problem that kills one-hot encoding on large datasets.

### Permutation-Based Feature Selection
After initial training, permutation importance identifies which features actually move the needle. A second model is trained on the reduced feature set and compared — the best performer is saved.

### IQR-Based Outlier Detection
Prices are filtered using the interquartile range instead of a fixed cap, adapting automatically to the distribution of any dataset.

### Preprocessing Cache
Processed features are serialised to disk with `joblib`. Subsequent runs skip preprocessing entirely — critical for fast iteration during experimentation.

---

## Usage

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost joblib optuna
```

### Run

```bash
python used_car_price_prediction.py
```

The script will search for a CSV file automatically. It looks for files matching:
- `regression of used car prices`
- `used car`
- `car price`

Or enter the path manually when prompted.

### Expected CSV Columns

| Column | Description |
|--------|-------------|
| `price` | Target variable (USD) |
| `model_year` | Year of manufacture |
| `milage` | Odometer reading |
| `brand` | Manufacturer |
| `model` | Model name |
| `engine` | Engine description (e.g. "300HP 3.5L V6") |
| `fuel_type` | Gas / Electric / Hybrid / Diesel |
| `transmission` | Automatic / Manual |
| `accident` | Accident history |
| `clean_title` | Yes / No |

### Outputs

| File | Description |
|------|-------------|
| `car_price_model.joblib` | Serialised model for inference |
| `feature_importance.csv` | Ranked feature importances |
| `model_evaluation/feature_importance.png` | Feature importance chart |
| `preprocessed_features.pkl` | Cached preprocessing (speeds up re-runs) |

---

## Making Predictions on New Data

```python
import joblib
import pandas as pd

model = joblib.load('car_price_model.joblib')

new_car = pd.DataFrame([{
    'model_year': 2020,
    'milage': 35000,
    'brand': 'Toyota',
    'model': 'Camry',
    'engine': '203HP 2.5L 4 Cylinder',
    'fuel_type': 'Gasoline',
    'transmission': 'Automatic',
    'accident': 'No',
    'clean_title': 'Yes'
}])

predicted_price = model.predict(new_car)
print(f"Predicted price: ${predicted_price[0]:,.2f}")
```

---

## Project Structure

```
Used-Cars-/
├── used_car_price_prediction.py   # Full ML pipeline
└── README.md
```

---

## License

MIT
