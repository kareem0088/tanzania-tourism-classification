"""
src/config.py
==============
Module Specifications:
- Define global configuration paths, data directories, and 
  column definitions (including numeric, categorical, ID, and 
  target columns).
- Centralize all hyperparameters and project settings to maintain
  consistency across training, evaluation, and feature engineering
  pipelines.
- Note: This script must be executed first to initialize,
  build, and save the required data files and directories
  before running the pipeline.
"""

from pathlib import Path
from typing import Dict, Any, List
from sqlalchemy import create_engine, text
import mlflow

'''
the following code sets up the directory structure for the project,
 including paths for raw data, features, processed data, models.
 It also defines constants for file paths,
 column names, and model parameters.
'''

BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
FEATURES_DIR: Path = DATA_DIR / "features"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = BASE_DIR / "models"

for directory in [RAW_DATA_DIR, FEATURES_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

VARIABLE_DEFINITIONS_PATH = DATA_DIR / 'VariableDefinitions.csv'
TRAIN_DATA_PATH: Path = RAW_DATA_DIR / "Train.csv"
TEST_DATA_PATH: Path = RAW_DATA_DIR / "Test.csv"
TRAIN_DATA_FE_PATH: Path = FEATURES_DIR / "train_data_FE.parquet"
TEST_DATA_FE_PATH: Path = FEATURES_DIR / "test_data_FE.parquet"
PROCESSED_TRAIN_PATH: Path = PROCESSED_DATA_DIR / "train_processed.parquet"
MODEL_ARTIFACT_PATH: Path = MODELS_DIR / "tourism_pipeline_bundle.joblib"
SUBMISSION_PATH: Path = PROCESSED_DATA_DIR / "submission.csv"

ID_COL: str = "Tour_ID"
TARGET_COL: str = "cost_category"
RANDOM_SEED: int = 42
N_SPLITS: int = 6
TEST_SIZE: float = 0.2


"""
-the following lists contain the base numeric and categorical columns used in the project.
-the base numeric columns are the original numeric features, while the base categorical 
 columns are the original categorical features.
-the base columns are used for data preprocessing, feature engineering, and model training in base model.
-other derived features are created based on these base columns during feature engineering.
-columns in NUMERIC_COLS and CATEGORICAL_COLS have been feature engineered and transformed,
while BASE_NUMERIC_COLS and BASE_CATEGORICAL_COLS remain unchanged.
"""



BASE_NUMERIC_COLS: list[str] = [
    "total_female",
    "total_male",
    "night_mainland",
    "night_zanzibar"
]

NUMERIC_COLS: List[str] = [
    "total_female",
    "total_male",
    "night_mainland",
    "night_zanzibar",
    # new features
    "total nights",
    "total_adults",
    "night_ratio",
    "adults_per_night",
    "male_to_female_ratio",



]

BASE_CATEGORICAL_COLS: List[str] = [
    "country",
    "age_group",
    "travel_with",
    "first_trip_tz",
    "purpose",
    "main_activity",
    "info_source",
    "tour_arrangement",
    "package_transport_int",
    "package_accomodation",
    "package_food",
    "package_transport_tz",
    "package_sightseeing",
    "package_guided_tour",
    "package_insurance",
    ]

CATEGORICAL_COLS: List[str] = [
    "country",
    "age_group",
    "travel_with",
    "first_trip_tz",
    "purpose",
    "main_activity",
    "info_source",
    "tour_arrangement",
    "package_transport_int",
    "package_accomodation",
    "package_food",
    "package_transport_tz",
    "package_sightseeing",
    "package_guided_tour",
    "package_insurance",
    # new features
    "continent"
    ]

NUMERIC_FEATURES: List[str] = NUMERIC_COLS
BASE_NUMERIC_FEATURES: List[str] = BASE_NUMERIC_COLS

CATEGORICAL_FEATURES: List[str] = CATEGORICAL_COLS
BASE_CATEGORICAL_FEATURES: List[str] = BASE_CATEGORICAL_COLS

TARGET_CLASSES: List[str] = [
    "Lower Cost",
    "Low Cost",
    "Normal Cost",
    "High Cost",
    "Higher Cost",
    "Highest Cost",
]

'''
the following dictionary contains the hyperparameters 
for the base model used in the project.
'''
BASE_MODEL_PARAMS: Dict[str, Any] ={
    'max_depth': 5,
    'nthread': 4,
    'eval_metric': 'mlogloss',
    'objective': 'multi:softprob',
    'num_class': len(TARGET_CLASSES),
    'random_state': RANDOM_SEED,
    'max_leaves': 15,
    'learning_rate': 0.01
 }


'''  
the following code sets up the SQLite database for MLflow tracking.
 It checks if the database file exists, creates it if it doesn't, and 
 sets the MLflow tracking URI to point to this SQLite database. This 
 allows you to log and track your machine learning experiments using MLflow.
'''
DB_FILE_PATH = BASE_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{DB_FILE_PATH}"

def create_sqlite_database_and_mlflow_link() -> str:
    print(f"- Checking SQLite database at: {DB_FILE_PATH}")
    
    engine = create_engine(MLFLOW_TRACKING_URI, echo=False)
    
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("- SQLite database file generated and connected.")
    except Exception as e:
        print(f"- Failed to connect or create SQLite database: {e}")
        raise e

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    return MLFLOW_TRACKING_URI

if __name__ == "__main__":
    create_sqlite_database_and_mlflow_link()
    # run "mlflow ui --backend-store-uri sqlite:///mlflow.db" in terminal to start the MLflow UI and access it at http://127.0.0.1:5000/