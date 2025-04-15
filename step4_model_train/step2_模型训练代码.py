import pandas as pd
import numpy as np
import glob
import joblib # For saving sklearn models (RF)
import logging
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer # For NaN handling
import lightgbm as lgb
import xgboost as xgb
# Ensure deep-forest is installed: pip install deep-forest
# It might require specific C++ build tools depending on your system
try:
    from deepforest.cascade import CascadeForestRegressor
    DEEP_FOREST_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Deep Forest library not found or import failed: {e}. Skipping Deep Forest training.")
    DEEP_FOREST_AVAILABLE = False


# ==============================================================================
# 1. Configuration
# ==============================================================================
YEAR = 2018
BASE_DIR = Path(r"E:\地理所\论文\中国XCO2论文_2025.04")
INPUT_CSV_DIR = BASE_DIR / "Extracted_Features_CSV" / str(YEAR)
OUTPUT_DIR = BASE_DIR / "处理结果"
MODEL_SAVE_DIR = OUTPUT_DIR / "模型数据"
LOG_DIR = OUTPUT_DIR / "模型日志"

# --- Create output directories if they don't exist ---
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- Target Variable ---
# !! IMPORTANT: Verify this is the correct column name for XCO2 !!
TARGET_COLUMN = 'xco2'

# --- Train/Test Split ---
TEST_SIZE = 0.2 # 20% for testing
RANDOM_STATE = 42 # For reproducibility

# --- Model Parameters (Start with defaults or simple settings) ---
# Add more specific hyperparameters as needed
LGBM_PARAMS = {
    'objective': 'regression_l1', # MAE loss, often robust
    'metric': 'rmse',
    'n_estimators': 1000, # Can increase, use with early stopping
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1, # Use all available CPU cores for data loading etc.
    'seed': RANDOM_STATE,
    'boosting_type': 'gbdt',
    # --- GPU Settings ---
    'device': 'gpu',
    'gpu_platform_id': 0, # Adjust if you have multiple platforms
    'gpu_device_id': 0   # Adjust if you have multiple GPUs
}

XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 1000, # Can increase, use with early stopping
    'learning_rate': 0.05,
    'max_depth': 7, # Typical starting point
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': RANDOM_STATE,
    'n_jobs': -1, # Use all available CPU cores for data loading etc.
    # --- GPU Settings ---
    'tree_method': 'hist', # Faster algorithm suitable for GPU
    'device': 'cuda'      # Use 'cuda' for NVIDIA GPUs
}

RF_PARAMS = {
    'n_estimators': 200, # Number of trees
    'max_depth': None, # Grow trees fully (can tune later)
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 0.6, # Consider sqrt or a fraction of features
    'random_state': RANDOM_STATE,
    'n_jobs': -1, # Use all available CPU cores
    'verbose': 1 # Show some progress
}

DF_PARAMS = { # Deep Forest - uses default parameters often
    'n_estimators': 2,  # Number of estimators in each cascade level's Random Forest layer
    'n_trees': 100,     # Number of trees in each estimator's Random Forest
    'max_layers': 10,   # Maximum number of cascade layers
    'random_state': RANDOM_STATE,
    'n_jobs': -1,       # Use all available CPU cores
    'verbose': 1
    # Add other params like min_samples_leaf if needed
}


# ==============================================================================
# 2. Logging Setup
# ==============================================================================
def setup_logging(log_file):
    """Configures logging to console and file."""
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # --- File Handler ---
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # --- Console Handler ---
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    return root_logger

# --- Setup logger for the main script run ---
timestamp = time.strftime("%Y%m%d_%H%M%S")
main_log_file = LOG_DIR / f"training_log_{timestamp}.log"
logger = setup_logging(main_log_file)

logger.info("Script started.")
logger.info(f"Input CSV Directory: {INPUT_CSV_DIR}")
logger.info(f"Model Save Directory: {MODEL_SAVE_DIR}")
logger.info(f"Log Directory: {LOG_DIR}")


# ==============================================================================
# 3. Data Loading and Preprocessing
# ==============================================================================
def load_and_preprocess_data(csv_dir, target_column, test_size, random_state, logger):
    """Loads data from monthly CSVs, preprocesses, and splits."""
    logger.info(f"Loading data for year {YEAR} from {csv_dir}...")
    all_files = list(csv_dir.glob("Extracted_Features_*.csv"))
    if not all_files:
        logger.error(f"No CSV files found in {csv_dir}. Exiting.")
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    df_list = []
    for f in all_files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            logger.warning(f"Could not read file {f}: {e}")
    
    if not df_list:
        logger.error("No data loaded after attempting to read CSV files. Exiting.")
        raise ValueError("No data loaded.")

    full_df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Combined data shape: {full_df.shape}")

    # --- Handle NaNs in Target ---
    initial_rows = len(full_df)
    full_df.dropna(subset=[target_column], inplace=True)
    rows_after_dropna_y = len(full_df)
    if initial_rows > rows_after_dropna_y:
        logger.warning(f"Dropped {initial_rows - rows_after_dropna_y} rows due to NaN in target column '{target_column}'.")
    
    if rows_after_dropna_y == 0:
        logger.error(f"No valid data remaining after dropping NaNs in target column '{target_column}'. Exiting.")
        raise ValueError(f"No valid data for target '{target_column}'.")


    # --- Identify Features (X) and Target (y) ---
    feature_columns = [col for col in full_df.columns if col not in ['fid', target_column]]
    logger.info(f"Using {len(feature_columns)} features: {feature_columns}")
    logger.info(f"Using target variable: {target_column}")

    X = full_df[feature_columns]
    y = full_df[target_column]

    # --- Split Data ---
    logger.info(f"Splitting data: Test size={test_size}, Random State={random_state}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train data shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

    # --- Handle NaNs in Features (Imputation) ---
    logger.info("Handling NaNs in features using Median Imputation...")
    imputer = SimpleImputer(strategy='median')
    # Fit imputer ONLY on training data
    X_train_imputed = imputer.fit_transform(X_train)
    # Transform both training and testing data
    X_test_imputed = imputer.transform(X_test)
    
    # Convert back to DataFrame to keep column names (optional but good practice)
    X_train = pd.DataFrame(X_train_imputed, columns=feature_columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_imputed, columns=feature_columns, index=X_test.index)
    
    # Check if NaNs remain (should not if imputer worked)
    nan_check_train = X_train.isnull().sum().sum()
    nan_check_test = X_test.isnull().sum().sum()
    logger.info(f"NaNs remaining in X_train after imputation: {nan_check_train}")
    logger.info(f"NaNs remaining in X_test after imputation: {nan_check_test}")
    if nan_check_train > 0 or nan_check_test > 0:
        logger.warning("NaN values still present after imputation. Check data or imputation strategy.")


    return X_train, X_test, y_train, y_test, feature_columns

# ==============================================================================
# 4. Model Training Functions
# ==============================================================================

def evaluate_model(model_name, y_true, y_pred, logger):
    """Calculates and logs R2 and RMSE."""
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    logger.info(f"{model_name} Evaluation: R2 = {r2:.4f}, RMSE = {rmse:.4f}")
    return r2, rmse

# --- LightGBM Training ---
def train_lightgbm(X_train, X_test, y_train, y_test, feature_names, model_save_path, logger):
    logger.info("--- Training LightGBM ---")
    start_time = time.time()
    
    # Add feature names for better logging/interpretation if needed
    # X_train.columns = feature_names
    # X_test.columns = feature_names
    
    try:
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        logger.info(f"LightGBM Parameters: {model.get_params()}") # Log effective parameters

        # Use eval_set for early stopping
        callbacks = [
            lgb.log_evaluation(period=100), # Log eval results every 100 iterations
            lgb.early_stopping(stopping_rounds=50, verbose=True) # Stop if no improvement for 50 rounds
        ]
        
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  eval_metric='rmse', # Metric for early stopping
                  callbacks=callbacks)

        end_time = time.time()
        logger.info(f"LightGBM training completed in {end_time - start_time:.2f} seconds.")

        # --- Evaluation ---
        logger.info("Evaluating LightGBM model...")
        y_pred_test = model.predict(X_test)
        evaluate_model("LightGBM", y_test, y_pred_test, logger)

        # --- Save Model ---
        logger.info(f"Saving LightGBM model to {model_save_path}")
        # Using joblib might be more consistent if switching between CPU/GPU later
        joblib.dump(model, model_save_path)
        # Or use native saving:
        # model.booster_.save_model(str(model_save_path).replace('.joblib', '.txt'))
        logger.info("LightGBM model saved.")

    except Exception as e:
        logger.error(f"Error during LightGBM training or saving: {e}", exc_info=True)


# --- XGBoost Training ---
def train_xgboost(X_train, X_test, y_train, y_test, feature_names, model_save_path, logger):
    logger.info("--- Training XGBoost ---")
    start_time = time.time()
    
    try:
        model = xgb.XGBRegressor(**XGB_PARAMS)
        logger.info(f"XGBoost Parameters: {model.get_params()}") # Log effective parameters

        # Use eval_set for early stopping
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  early_stopping_rounds=50, # Stop if no improvement for 50 rounds
                  verbose=100) # Log eval results every 100 iterations

        end_time = time.time()
        logger.info(f"XGBoost training completed in {end_time - start_time:.2f} seconds.")

        # --- Evaluation ---
        logger.info("Evaluating XGBoost model...")
        y_pred_test = model.predict(X_test)
        evaluate_model("XGBoost", y_test, y_pred_test, logger)

        # --- Save Model ---
        logger.info(f"Saving XGBoost model to {model_save_path}")
        # Native saving is generally recommended for XGBoost
        model.save_model(model_save_path)
        # Or use joblib: joblib.dump(model, model_save_path)
        logger.info("XGBoost model saved.")

    except Exception as e:
        logger.error(f"Error during XGBoost training or saving: {e}", exc_info=True)


# --- Random Forest Training ---
def train_random_forest(X_train, X_test, y_train, y_test, feature_names, model_save_path, logger):
    logger.info("--- Training Random Forest ---")
    start_time = time.time()
    
    try:
        model = RandomForestRegressor(**RF_PARAMS)
        logger.info(f"Random Forest Parameters: {model.get_params()}")

        # Scikit-learn RF runs on CPU (n_jobs=-1 utilizes cores)
        model.fit(X_train, y_train)

        end_time = time.time()
        logger.info(f"Random Forest training completed in {end_time - start_time:.2f} seconds.")

        # --- Evaluation ---
        logger.info("Evaluating Random Forest model...")
        y_pred_test = model.predict(X_test)
        evaluate_model("Random Forest", y_test, y_pred_test, logger)

        # --- Save Model ---
        logger.info(f"Saving Random Forest model to {model_save_path}")
        joblib.dump(model, model_save_path)
        logger.info("Random Forest model saved.")

    except Exception as e:
        logger.error(f"Error during Random Forest training or saving: {e}", exc_info=True)


# --- Deep Forest Training ---
def train_deep_forest(X_train, X_test, y_train, y_test, feature_names, model_save_path, logger):
    if not DEEP_FOREST_AVAILABLE:
        logger.warning("Skipping Deep Forest training as library is not available.")
        return
        
    logger.info("--- Training Deep Forest ---")
    start_time = time.time()
    
    try:
        model = CascadeForestRegressor(**DF_PARAMS)
        # Logging parameters for Deep Forest might require manual formatting if get_params isn't standard
        logger.info(f"Deep Forest Parameters: {DF_PARAMS}")

        # Deep Forest typically runs on CPU (n_jobs=-1 utilizes cores)
        # It often expects numpy arrays
        X_train_np = X_train.to_numpy()
        y_train_np = y_train.to_numpy()
        X_test_np = X_test.to_numpy()

        model.fit(X_train_np, y_train_np)

        end_time = time.time()
        logger.info(f"Deep Forest training completed in {end_time - start_time:.2f} seconds.")

        # --- Evaluation ---
        logger.info("Evaluating Deep Forest model...")
        y_pred_test = model.predict(X_test_np)
        evaluate_model("Deep Forest", y_test, y_pred_test, logger)

        # --- Save Model ---
        logger.info(f"Saving Deep Forest model to {model_save_path}")
        # Use Deep Forest's specific save method if available (check docs)
        # Placeholder using joblib, but native method is preferred if exists
        try:
            model.save(str(model_save_path).replace('.joblib','')) # df typically saves a directory
            logger.info("Deep Forest model saved using native save.")
        except AttributeError:
            logger.warning("Deep Forest model does not have a native '.save()' method in this version? Trying joblib.")
            joblib.dump(model, model_save_path) # Fallback to joblib
            logger.info("Deep Forest model saved using joblib.")


    except Exception as e:
        logger.error(f"Error during Deep Forest training or saving: {e}", exc_info=True)


# ==============================================================================
# 5. Main Execution
# ==============================================================================
if __name__ == "__main__":
    logger.info("="*30 + " Starting Model Training Process " + "="*30)

    # --- Load Data ---
    try:
        X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(
            INPUT_CSV_DIR, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, logger
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load data: {e}. Stopping execution.")
        exit() # Exit if data loading fails critically


    # --- Train Models ---
    # Define model save paths
    lgbm_model_path = MODEL_SAVE_DIR / f"lightgbm_model_{YEAR}_{timestamp}.joblib"
    xgb_model_path = MODEL_SAVE_DIR / f"xgboost_model_{YEAR}_{timestamp}.json" # Native format often json/ubj/bin
    rf_model_path = MODEL_SAVE_DIR / f"randomforest_model_{YEAR}_{timestamp}.joblib"
    df_model_path = MODEL_SAVE_DIR / f"deepforest_model_{YEAR}_{timestamp}.joblib" # Or a directory if native save used

    # Train LightGBM
    train_lightgbm(X_train, X_test, y_train, y_test, feature_names, lgbm_model_path, logger)
    logger.info("-" * 60)

    # Train XGBoost
    train_xgboost(X_train, X_test, y_train, y_test, feature_names, xgb_model_path, logger)
    logger.info("-" * 60)

    # Train Random Forest
    train_random_forest(X_train, X_test, y_train, y_test, feature_names, rf_model_path, logger)
    logger.info("-" * 60)

    # Train Deep Forest
    train_deep_forest(X_train, X_test, y_train, y_test, feature_names, df_model_path, logger)
    logger.info("-" * 60)


    logger.info("="*30 + " Model Training Process Finished " + "="*30)