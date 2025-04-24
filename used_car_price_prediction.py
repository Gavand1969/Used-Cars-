"""
# Advanced Stacked Car Price Regression Model
# 
# A high-performance model featuring:
# - Target encoding for categorical features
# - Stacked ensemble with LightGBM, XGBoost, and CatBoost
# - Permutation-based feature selection
# - Preprocessing caching for faster experimentation
#
# Date: April 22, 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import StackingRegressor
import lightgbm as lgb
from lightgbm import early_stopping
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# Define CarPriceModel at the global level to allow pickling
class CarPriceModel:
    def __init__(self, model, encoders=None, log_transform=True):
        self.model = model
        self.encoders = encoders or {}
        self.log_transform = log_transform
    
    def predict(self, X):
        # Apply encoding if needed
        X_processed = self._preprocess(X)
        
        # Make prediction
        preds_raw = self.model.predict(X_processed)
            
        # Convert from log if needed
        if self.log_transform:
            return np.expm1(preds_raw)
        return preds_raw
    
    def _preprocess(self, X):
        """Apply preprocessing to new data"""
        X_copy = X.copy()
        
        # Apply target encoders to categorical columns
        for col, encoder in self.encoders.items():
            if col in X_copy.columns:
                # Get global mean for unknown categories
                global_mean = encoder['__global_mean__']
                
                # Apply encoding with fallback to global mean
                X_copy[col] = X_copy[col].map(lambda x: encoder.get(x, global_mean))
        
        return X_copy

# Try to import required libraries or install them
def ensure_libraries():
    """Ensure all required libraries are available"""
    libraries = {
        'lightgbm': 'lightgbm',
        'xgboost': 'xgboost',
        'catboost': 'catboost',
        'optuna': 'optuna'
    }
    
    missing_libs = []
    
    for module, package in libraries.items():
        try:
            __import__(module)
            print(f"✓ {module} is available")
        except ImportError:
            missing_libs.append(package)
            print(f"✗ {module} not found, will try to install")
    
    if missing_libs:
        try:
            import sys
            import subprocess
            for lib in missing_libs:
                print(f"Installing {lib}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
                print(f"Successfully installed {lib}")
        except Exception as e:
            print(f"Error installing libraries: {e}")
            print("Some optimizations may not be available")
    
    # Now try to import everything again
    try:
        import xgboost as xgb
        XGBOOST_AVAILABLE = True
        print("XGBoost is available for ensemble stacking")
    except ImportError:
        XGBOOST_AVAILABLE = False
        print("XGBoost not available")
    
    try:
        import catboost
        CATBOOST_AVAILABLE = True
        print("CatBoost is available for ensemble stacking")
    except ImportError:
        CATBOOST_AVAILABLE = False
        print("CatBoost not available")
    
    try:
        import optuna
        OPTUNA_AVAILABLE = True
        print("Optuna is available for hyperparameter optimization")
    except ImportError:
        OPTUNA_AVAILABLE = False
        print("Optuna not available")
    
    return XGBOOST_AVAILABLE, CATBOOST_AVAILABLE, OPTUNA_AVAILABLE

# Set global flags
XGBOOST_AVAILABLE, CATBOOST_AVAILABLE, OPTUNA_AVAILABLE = False, False, False

# Find the CSV file in Google Colab
def find_csv_file(pattern):
    """Find CSV file in current directory with a matching pattern"""
    print(f"Searching for CSV files matching '{pattern}'...")
    found_files = []
    
    # Check current directory first
    for file in os.listdir():
        if file.endswith('.csv') and pattern.lower() in file.lower():
            found_files.append(file)
    
    # Check content directory (common in Colab)
    content_dir = "/content"
    if os.path.exists(content_dir):
        for file in os.listdir(content_dir):
            if file.endswith('.csv') and pattern.lower() in file.lower():
                found_files.append(os.path.join(content_dir, file))
    
    # Check sample_data directory (common in Colab)
    sample_dir = "/content/sample_data"
    if os.path.exists(sample_dir):
        for file in os.listdir(sample_dir):
            if file.endswith('.csv') and pattern.lower() in file.lower():
                found_files.append(os.path.join(sample_dir, file))
    
    # Check drive directory if mounted
    drive_dir = "/content/drive/MyDrive"
    if os.path.exists(drive_dir):
        for root, dirs, files in os.walk(drive_dir):
            for file in files:
                if file.endswith('.csv') and pattern.lower() in file.lower():
                    found_files.append(os.path.join(root, file))
    
    if found_files:
        print(f"Found {len(found_files)} matching CSV files:")
        for i, file in enumerate(found_files):
            print(f"{i+1}. {file}")
        
        if len(found_files) == 1:
            print(f"Automatically selecting the only match: {found_files[0]}")
            return found_files[0]
        else:
            selection = input(f"Enter the number (1-{len(found_files)}) of the file to use: ")
            try:
                index = int(selection) - 1
                if 0 <= index < len(found_files):
                    return found_files[index]
            except:
                pass
            print("Invalid selection. Using the first file.")
            return found_files[0]
    
    print(f"No matching CSV file found for pattern '{pattern}'")
    return None

def target_encode(df, col, target, smoothing=10, min_samples=5):
    """
    Apply target encoding to a categorical feature - FIXED VERSION
    
    Args:
        df: DataFrame containing the data
        col: Name of the categorical column to encode
        target: Name or Series of the target variable
        smoothing: Smoothing factor for regularization
        min_samples: Minimum samples required for a category
        
    Returns:
        encoder: Dictionary mapping category to encoded value
    """
    # Make sure the target is a Series or numpy array
    if isinstance(target, str):
        y = df[target]
    else:
        y = target
    
    # Calculate global mean
    global_mean = float(y.mean())
    
    # Create encoder dictionary
    encoder = {}
    encoder['__global_mean__'] = global_mean
    
    # Handle missing values in the categorical column
    # Fill missing values with a placeholder
    df_temp = df.copy()
    df_temp[col] = df_temp[col].fillna('__MISSING__')
    
    # Create a Series for the target if it's not already
    if not isinstance(y, pd.Series):
        y_series = pd.Series(y, index=df_temp.index)
    else:
        y_series = y
    
    # Group by category and calculate mean and count
    try:
        # Make a temporary DataFrame with just the category and target
        temp_df = pd.DataFrame({
            'category': df_temp[col],
            'target': y_series
        })
        
        # Group by category
        grouped = temp_df.groupby('category')['target'].agg(['mean', 'count'])
        
        # Process each group
        for category, (mean, count) in zip(grouped.index, grouped.values):
            # Apply smoothing based on count
            if count >= min_samples:
                weight = count / (count + smoothing)
                regularized_mean = weight * mean + (1 - weight) * global_mean
            else:
                regularized_mean = global_mean
                
            # Store in encoder
            encoder[category] = float(regularized_mean)
        
        # Handle any categories that might have been missed
        for category in df_temp[col].unique():
            if category not in encoder:
                encoder[category] = float(global_mean)
                
    except Exception as e:
        print(f"Error in target encoding: {e}")
        # If there's an error, return a simple encoder with global mean
        for category in df_temp[col].unique():
            encoder[category] = float(global_mean)
    
    return encoder

def load_and_prepare_data(data_path, cache_file='preprocessed_features.pkl', price_cap=None, log_transform=True):
    """
    Load and prepare data with advanced feature engineering and target encoding
    
    Args:
        data_path: Path to the CSV file
        cache_file: Path to cache processed features
        price_cap: Cap for prices based on IQR (set to None to use IQR-based outlier detection)
        log_transform: Whether to apply log transformation to price
        
    Returns:
        X: Feature matrix
        y: Target vector
        df: Processed DataFrame
        encoders: Dictionary of target encoders
        log_transform: Whether log transform was applied
    """
    # Check if cached features exist
    if os.path.exists(cache_file):
        print(f"Loading preprocessed features from {cache_file}")
        cached_data = joblib.load(cache_file)
        return (cached_data['X'], cached_data['y'], cached_data['df'], 
                cached_data['encoders'], cached_data['log_transform'])
    
    start_time = time.time()
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records with {len(df.columns)} features")
    print(f"Columns: {', '.join(df.columns)}")
    
    print("\nPerforming feature engineering...")
    
    # 1. Extract horsepower from engine column
    def extract_horsepower(engine_str):
        try:
            if pd.isna(engine_str): 
                return np.nan
            if isinstance(engine_str, str) and 'HP' in engine_str:
                return float(engine_str.split('HP')[0].strip())
            return np.nan
        except:
            return np.nan
    
    if 'engine' in df.columns:
        df['horsepower'] = df['engine'].apply(extract_horsepower)
        # Fill missing values with median
        median_hp = df['horsepower'].median()
        df['horsepower'] = df['horsepower'].fillna(median_hp)
        # Add non-linear transformations of horsepower
        df['horsepower_squared'] = df['horsepower'] ** 2
        df['horsepower_log'] = np.log1p(df['horsepower'])
    
    # 2. Calculate car age with transformations
    if 'model_year' in df.columns:
        # Fill missing model years with median
        median_year = df['model_year'].median()
        df['model_year'] = df['model_year'].fillna(median_year)
        
        current_year = 2025
        df['car_age'] = current_year - df['model_year']
        
        # Add non-linear transformations for age
        df['car_age_squared'] = df['car_age'] ** 2
        df['car_age_log'] = np.log1p(df['car_age'])
        
        # Exponential decay to model depreciation curve
        df['depreciation_factor'] = np.exp(-0.15 * df['car_age'])
    
    # 3. Create binary features
    if 'accident' in df.columns:
        # Handle NA values explicitly by filling with 'No'
        df['had_accident'] = df['accident'].fillna('No').apply(
            lambda x: 0 if x == 'No' or x == '0' else 1)
    
    if 'clean_title' in df.columns:
        # Handle NA values explicitly by filling with 'No'
        df['is_clean_title'] = df['clean_title'].fillna('No').apply(
            lambda x: 1 if x == 'Yes' else 0)
    
    # 4. Enhanced mileage features
    if 'milage' in df.columns:
        # Fill missing mileage values with median
        median_milage = df['milage'].median()
        df['milage'] = df['milage'].fillna(median_milage)
        
        # Log transformation
        df['log_milage'] = np.log1p(df['milage'])
        
        # Ensure car_age has no zeros before division
        car_age_for_division = df['car_age'].replace(0, 0.5)
        
        # Miles per year
        df['miles_per_year'] = df['milage'] / car_age_for_division
        
        # Log transformation of miles per year
        df['log_miles_per_year'] = np.log1p(df['miles_per_year'])
        
        # Mileage relative to expected (average car does ~12K miles/year)
        expected_milage = df['car_age'] * 12000
        df['milage_vs_expected'] = df['milage'] / expected_milage.replace(0, 1)
        df['high_milage_for_age'] = (df['milage_vs_expected'] > 1.2).astype(int)
        df['low_milage_for_age'] = (df['milage_vs_expected'] < 0.8).astype(int)
    
    # 5. Use IQR-based outlier detection for prices instead of fixed cap
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Filter out outliers (1.5 * IQR rule for upper bound, lower bound at $100)
    original_size = len(df)
    upper_bound = Q3 + 1.5 * IQR if price_cap is None else price_cap
    df = df[(df['price'] <= upper_bound) & (df['price'] > 100)]
    removed = original_size - len(df)
    print(f"Removed {removed} records with price > ${upper_bound:,.2f} or < $100 (IQR-based filtering)")
    
    # 6. Enhanced fuel type features
    if 'fuel_type' in df.columns:
        # Fill missing values with most common fuel type
        most_common_fuel = df['fuel_type'].mode()[0]
        df['fuel_type'] = df['fuel_type'].fillna(most_common_fuel)
        
        # Create binary features for common fuel types
        for fuel in ['electric', 'hybrid', 'diesel', 'gas']:
            df[f'is_{fuel}'] = df['fuel_type'].str.lower().str.contains(fuel, na=False).astype(int)
    
    # 7. Create simplified model field
    if 'model' in df.columns:
        # Extract model base name (e.g., "Civic" from "Honda Civic LX")
        def extract_base_model(model_str):
            if pd.isna(model_str):
                return 'Unknown'
            parts = str(model_str).split()
            return parts[0] if len(parts) == 1 else parts[0] + " " + parts[1]
        
        df['model_base'] = df['model'].apply(extract_base_model)
    
    # 8. Transmission features
    if 'transmission' in df.columns:
        # Fill missing values
        df['transmission'] = df['transmission'].fillna('unknown')
        
        # Create binary feature for automatic transmission
        df['is_automatic'] = df['transmission'].str.lower().str.contains('auto', na=False).astype(int)
    
    # Handle missing values for all columns
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype == 'object':
                # For categorical columns, fill with most frequent value
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
            else:
                # For numerical columns, fill with median
                df[col] = df[col].fillna(df[col].median())
    
    # Create target variable early (for use in target encoding)
    if log_transform:
        print("Applying log transformation to price")
        y_value = np.log1p(df['price'])  # log(1+x) to handle zero prices
    else:
        y_value = df['price']
    
    # Identify categorical columns for target encoding
    categorical_cols = []
    for col in df.columns:
        if col != 'price' and df[col].dtype == 'object':
            categorical_cols.append(col)
    
    print(f"Applying target encoding to {len(categorical_cols)} categorical features")
    
    # Target encode categorical features
    encoders = {}
    
    for col in categorical_cols:
        # Skip columns with too many unique values (may cause memory issues)
        n_unique = df[col].nunique()
        if n_unique > 1000:
            print(f"Skipping target encoding for {col} ({n_unique} unique values)")
            continue
            
        try:
            print(f"Target encoding {col} ({n_unique} unique values)")
            encoder = target_encode(df, col, y_value)
            encoders[col] = encoder
            
            # Apply encoding safely
            df[col] = df[col].map(lambda x: encoder.get(x, encoder['__global_mean__']))
        except Exception as e:
            print(f"Error during target encoding of {col}: {e}")
            # Skip this column if encoding fails
            continue
    
    # Special handling for high-cardinality columns
    high_cardinality_cols = ['brand', 'model', 'model_base']
    for col in high_cardinality_cols:
        if col in df.columns and col not in encoders:
            try:
                print(f"Target encoding {col} ({df[col].nunique()} unique values)")
                encoder = target_encode(df, col, y_value)
                encoders[col] = encoder
                
                # Apply encoding safely
                df[col] = df[col].map(lambda x: encoder.get(x, encoder['__global_mean__']))
            except Exception as e:
                print(f"Error during target encoding of {col}: {e}")
                # Skip this column if encoding fails
                continue
    
    # Prepare for the model
    print("\nPreparing feature matrix and target vector...")
    
    # Select columns to drop
    drop_cols = ['price']
    for col in ['id', 'engine', 'ext_col', 'int_col', 'accident', 'clean_title']:
        if col in df.columns:
            drop_cols.append(col)
    
    # Create feature matrix
    X = df.drop(drop_cols, axis=1)
    
    # y_value was already created above
    y = y_value
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Feature engineering completed in {time.time() - start_time:.2f} seconds")
    
    # Save processed features for future runs
    try:
        joblib.dump({
            'X': X,
            'y': y,
            'df': df,
            'encoders': encoders,
            'log_transform': log_transform
        }, cache_file)
        print(f"Cached preprocessed features to {cache_file}")
    except Exception as e:
        print(f"Could not cache preprocessed features: {e}")
    
    return X, y, df, encoders, log_transform

def select_features_permutation(X, y, model, n_features=None, random_state=42):
    """
    Select features using permutation importance
    
    Args:
        X: Feature matrix
        y: Target vector
        model: Trained model
        n_features: Number of features to select (if None, selects features with importance > 0)
        random_state: Random seed
        
    Returns:
        selected_features: List of selected feature names
        importance_df: DataFrame with feature importances
    """
    print("\nPerforming permutation-based feature selection...")
    
    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X, y, n_repeats=10, random_state=random_state, n_jobs=-1
    )
    
    # Create DataFrame with feature importances
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Importance', ascending=False)
    
    # Select features with positive importance or top n_features
    if n_features is None:
        selected_features = importance_df[importance_df['Importance'] > 0]['Feature'].tolist()
    else:
        selected_features = importance_df.head(n_features)['Feature'].tolist()
    
    print(f"Selected {len(selected_features)} features based on permutation importance")
    
    # Print top 10 features
    print("\nTop 10 Features by Permutation Importance:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        print(f"{i+1}. {row['Feature']}: {row['Importance']:.4f} ± {row['Std']:.4f}")
    
    return selected_features, importance_df

def train_stacked_ensemble(X, y, test_size=0.2, random_state=42):
    """
    Train a stacked ensemble model with LightGBM, XGBoost, and CatBoost
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Validation set size
        random_state: Random seed
        
    Returns:
        ensemble: Trained ensemble model
        metrics: Dictionary of evaluation metrics
        feature_importance: Feature importance DataFrame
    """
    start_time = time.time()
    print("\nPreparing stacked ensemble model...")
    
    # Make sure we have required libraries
    global XGBOOST_AVAILABLE, CATBOOST_AVAILABLE, OPTUNA_AVAILABLE
    if not (XGBOOST_AVAILABLE or CATBOOST_AVAILABLE):
        XGBOOST_AVAILABLE, CATBOOST_AVAILABLE, OPTUNA_AVAILABLE = ensure_libraries()
    
    # Split the data for training
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} records, Validation set: {X_val.shape[0]} records")
    
    # Define base models for the ensemble
    base_models = []
    
    # Add LightGBM
    lgb_params = {
        'n_estimators': 2000,
        'max_depth': 12,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_child_samples': 20,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': random_state
    }
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    base_models.append(('lgb', lgb_model))
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        import xgboost as xgb
        xgb_params = {
            'n_estimators': 2000,
            'max_depth': 10,
            'learning_rate': 0.02,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'gamma': 0.1,
            'verbosity': 0,
            'n_jobs': -1,
            'random_state': random_state
        }
        xgb_model = xgb.XGBRegressor(**xgb_params)
        base_models.append(('xgb', xgb_model))
    
    # Add CatBoost if available
    if CATBOOST_AVAILABLE:
        from catboost import CatBoostRegressor
        cat_params = {
            'iterations': 2000,
            'depth': 10,
            'learning_rate': 0.02,
            'l2_leaf_reg': 3,
            'random_strength': 0.1,
            'verbose': 0,
            'thread_count': -1,
            'random_seed': random_state
        }
        cat_model = CatBoostRegressor(**cat_params)
        base_models.append(('cat', cat_model))
    
    # Define the meta-learner
    meta_learner = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        verbose=-1,
        n_jobs=-1,
        random_state=random_state
    )
    
    # Create the stacked ensemble
    ensemble = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    # Train the ensemble
    print("\nTraining stacked ensemble model...")
    print(f"Base models: {[model[0] for model in base_models]}")
    ensemble.fit(X_train, y_train)
    
    # Get predictions
    y_pred = ensemble.predict(X_val)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    # Try to get feature importance from the LightGBM model
    feature_importance = None
    if hasattr(ensemble.estimators_[0], 'feature_importances_'):
        lgb_model = ensemble.estimators_[0]
        importances = lgb_model.feature_importances_
        
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Features by LightGBM Importance:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"{i+1}. {row['Feature']}: {row['Importance']:.4f}")
    
    # Create metrics dictionary
    metrics = {
        'rmse': rmse,
        'r2': r2
    }
    
    return ensemble, metrics, feature_importance

def main():
    """Run the advanced stacked car price prediction pipeline"""
    print("======= Advanced Stacked Car Price Prediction =======")
    total_start_time = time.time()
    
    # Ensure required libraries are available
    global XGBOOST_AVAILABLE, CATBOOST_AVAILABLE, OPTUNA_AVAILABLE
    XGBOOST_AVAILABLE, CATBOOST_AVAILABLE, OPTUNA_AVAILABLE = ensure_libraries()
    
    # 1. Find and load data
    possible_patterns = [
        "regression of used car prices",
        "used car",
        "car price"
    ]
    
    data_path = None
    for pattern in possible_patterns:
        found_path = find_csv_file(pattern)
        if found_path:
            data_path = found_path
            break
    
    if not data_path:
        print("Could not find CSV file automatically. Please specify the filename:")
        data_path = input("Enter CSV filename (full path): ")
        if not os.path.exists(data_path):
            print("File not found. Exiting.")
            return
    
    # 2. Load and prepare data with target encoding
    try:
        X, y, df, encoders, log_transform = load_and_prepare_data(
            data_path, 
            price_cap=None,  # Use IQR-based outlier detection
            log_transform=True
        )
    except Exception as e:
        print(f"Error during data preparation: {e}")
        return
    
    # 3. Train stacked ensemble model
    try:
        ensemble, metrics, feature_importance = train_stacked_ensemble(
            X, y, test_size=0.2, random_state=42
        )
    except Exception as e:
        print(f"Error during model training: {e}")
        return
    
    # 4. Perform permutation-based feature selection
    try:
        selected_features, perm_importance = select_features_permutation(
            X, y, ensemble, n_features=30, random_state=42
        )
        
        # Train final model on selected features
        print("\nTraining final model with selected features...")
        X_selected = X[selected_features]
        final_ensemble, final_metrics, _ = train_stacked_ensemble(
            X_selected, y, test_size=0.2, random_state=42
        )
        
        # Compare performance
        print("\nPerformance Comparison:")
        print(f"Full model: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
        print(f"Selected features: R² = {final_metrics['r2']:.4f}, RMSE = {final_metrics['rmse']:.4f}")
        
        # Choose the best model
        if final_metrics['r2'] >= metrics['r2']:
            print("Using model with selected features (better performance)")
            final_model = final_ensemble
            final_X = X_selected
        else:
            print("Using full model (better performance)")
            final_model = ensemble
            final_X = X
    except Exception as e:
        print(f"Error during feature selection: {e}")
        print("Using full model without feature selection")
        final_model = ensemble
        final_X = X
    
    # 5. Create and save the final model
    try:
        model = CarPriceModel(final_model, encoders, log_transform)
        joblib.dump(model, 'car_price_model.joblib')
        print("\nFinal model saved to car_price_model.joblib")
        
        # Save feature importance
        if feature_importance is not None:
            feature_importance.to_csv('feature_importance.csv', index=False)
            print("Feature importance saved to feature_importance.csv")
        
        # Create simple visualizations
        os.makedirs("model_evaluation", exist_ok=True)
        
        # Feature importance plot
        plt.figure(figsize=(12, 10))
        importance_df = perm_importance if 'perm_importance' in locals() else feature_importance
        if importance_df is not None:
            top_n = min(20, len(importance_df))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
            plt.title('Top Features by Importance')
            plt.tight_layout()
            plt.savefig('model_evaluation/feature_importance.png', dpi=300)
            plt.close()
    except Exception as e:
        print(f"Error saving model: {e}")
    
    print("\n======= Pipeline Complete =======")
    print(f"Final R² Score: {final_metrics['r2'] if 'final_metrics' in locals() else metrics['r2']:.4f}")
    print(f"Total execution time: {(time.time() - total_start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    # Run the main pipeline
    main()

 