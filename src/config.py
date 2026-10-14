from pathlib import Path
from typing import Dict, Any, List

BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = BASE_DIR / "models"

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

TRAIN_DATA_PATH: Path = RAW_DATA_DIR / "Train.csv"
TEST_DATA_PATH: Path = RAW_DATA_DIR / "Test.csv"
PROCESSED_TRAIN_PATH: Path = PROCESSED_DATA_DIR / "train_processed.parquet"
MODEL_ARTIFACT_PATH: Path = MODELS_DIR / "tourism_pipeline_bundle.joblib"
SUBMISSION_PATH: Path = PROCESSED_DATA_DIR / "submission.csv"

ID_COL: str = "Tour_ID"
TARGET_COL: str = "cost_category"
RANDOM_SEED: int = 42
N_SPLITS: int = 5

NUMERIC_COLS: List[str] = [
    "total_female",
    "total_male",
    "night_mainland",
    "night_zanzibar",]

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
    "payment_mode",
]

TARGET_CLASSES: List[str] = [
    "Lower Cost",
    "Low Cost",
    "Normal Cost",
    "High Cost",
    "Higher Cost",
    "Highest Cost",
]