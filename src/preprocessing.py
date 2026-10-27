import numpy as np 
import pandas as pd
from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH, NUMERIC_COLS, CATEGORICAL_COLS, ID_COL, TARGET_COL

def load_data(file_path: str, expected_columns: str) -> pd.DataFrame:
    raw_df = pd.read_csv(file_path)
    if raw_df.shape[1] != expected_columns:
        raise ValueError("columns number unexpected")
    return raw_df

 
def datatypes(raw_df) -> pd.DataFrame:
    if ID_COL not in raw_df.columns:
        raise KeyError(f"{ID_COL}, is missing.")
    """
    if not (pd.api.types.is_object_dtype(raw_df[ID_COL])):
        raw_df[ID_COL] = raw_df[ID_COL].astype(object) 
    """
    for col in NUMERIC_COLS:
        if col not in raw_df.columns:
            raise KeyError(f"{col}, is missing.")
    """
        if not (pd.api.types.is_numeric_dtype(raw_df[col])):
            #raw_df[col] = raw_df[col].astype('int64')
    """
    for col in CATEGORICAL_COLS:
        if col not in raw_df.columns:
            raise KeyError(f"{col}, is missing.")
    """
        if not (pd.api.types.is_object_dtype(raw_df[col])
                or pd.api.types.is_string_dtype(raw_df[col])
                or ):
            raw_df[col] = raw_df[col].astype(object)
    """
    
    if TARGET_COL not in raw_df.columns:
        raise KeyError(f"{TARGET_COL}, is missing.")
    """   
    if not (pd.api.types.is_object_dtype(raw_df[TARGET_COL])):
        raw_df[TARGET_COL] = raw_df[TARGET_COL].astype(object)
    """
    return raw_df

def clean_duplicated(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.duplicated().sum() != 0:
        raw_df.drop_duplicates(inplace=True)
    return raw_df

def missing_alone(raw_df: pd.DataFrame) -> pd.DataFrame:
    # for alone values 
    mask_alone = (raw_df['total_female'] + raw_df['total_male']) <= 1 
    mask_nan = raw_df['travel_with'].isna()
    mask = mask_nan & mask_alone
    raw_df.loc[mask, 'travel_with'] = 'Alone'
    return raw_df

def missing_spouse(raw_df):
    mask_nan = raw_df['travel_with'].isna()
    mask_female = raw_df['total_female'] == 1  
    mask_male   = raw_df['total_male'] == 1 
    mask_purpose = raw_df['purpose'].isin(['Leisure and Holidays', 'Visiting Friends and Relatives'])

    mask = mask_nan & mask_female & mask_male & mask_purpose
    raw_df.loc[mask, 'travel_with'] = 'With Spouse' 
    return raw_df


if __name__ == '__main__':
    raw_df = load_data(TRAIN_DATA_PATH)
    raw_df = datatypes(raw_df)
    raw_df = clean_duplicated(raw_df)
    raw_df = missing_alone(raw_df)
    raw_df = missing_spouse(raw_df)