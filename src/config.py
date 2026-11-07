from pathlib import Path
from typing import Dict, Any, List

BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
FEATURES_DIR: Path = DATA_DIR / "features"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = BASE_DIR / "models"
MLFLOW_TRACKING_DIR: Path = BASE_DIR / "mlruns"

for directory in [RAW_DATA_DIR, FEATURES_DIR, PROCESSED_DATA_DIR, MODELS_DIR, MLFLOW_TRACKING_DIR]:
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

