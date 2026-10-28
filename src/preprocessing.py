import numpy as np 
import pandas as pd
from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH, NUMERIC_COLS, CATEGORICAL_COLS, ID_COL, TARGET_COL

class Data_Preprocessor:
    def __init__(self, expected_columns: int, is_train: bool = True ):
        self.expected_columns = expected_columns
        self.is_train = is_train

    def load_data(self, file_path: str) -> pd.DataFrame:
        raw_df = pd.read_csv(file_path)
        if raw_df.shape[1] != self.expected_columns:
            raise ValueError("columns number unexpected.")
        return raw_df

    def datatypes(self, raw_df) -> pd.DataFrame:
        if ID_COL not in raw_df.columns:
            raise KeyError(f"{ID_COL}, is missing.")
        
        for col in NUMERIC_COLS:
            if col not in raw_df.columns:
                raise KeyError(f"{col}, is missing.")
        
            if not (pd.api.types.is_integer_dtype(raw_df[col])):
                raw_df[col] = raw_df[col].astype('Int64')
        
        for col in CATEGORICAL_COLS:
            if col not in raw_df.columns:
                raise KeyError(f"{col}, is missing.")
            if pd.api.types.is_object_dtype(raw_df[col]):
                continue
            elif pd.api.types.is_string_dtype(raw_df[col]):
                raw_df[col] = raw_df[col].astype(object)
            elif isinstance(raw_df[col].dtype, pd.CategoricalDtype):
                raw_df[col] = raw_df[col].astype(object)
            else:
                raise KeyError("categorical columns is not object.")
        if self.is_train:
            if TARGET_COL not in raw_df.columns:
                raise KeyError("Target column is missing.") 
            if not (pd.api.types.is_object_dtype(raw_df[TARGET_COL])):
                raise KeyError("Target column should be object")
        else:
            if TARGET_COL in raw_df.columns:
                raise KeyError("Target column is in test dataset")

        return raw_df

    def clean_duplicated(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df.duplicated().sum() != 0:
            raw_df.drop_duplicates(inplace=True)
            raw_df.reset_index(drop=True, inplace=True)
        return raw_df

    def missing_alone(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        # for alone values 
        mask_alone = ((raw_df['total_female'] + raw_df['total_male']) <= 1).fillna(False)
        mask_nan = raw_df['travel_with'].isna()
        mask = mask_nan & mask_alone
        raw_df.loc[mask, 'travel_with'] = 'Alone'
        return raw_df

    def missing_spouse(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        mask_nan = raw_df['travel_with'].isna()
        mask_female = (raw_df['total_female'] == 1).fillna(False)  
        mask_male   = (raw_df['total_male'] == 1).fillna(False) 
        mask_purpose = raw_df['purpose'].isin(['Leisure and Holidays', 'Visiting Friends and Relatives'])

        mask = mask_nan & mask_female & mask_male & mask_purpose
        raw_df.loc[mask, 'travel_with'] = 'With Spouse' 
        return raw_df

    def drop_missing(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if self.is_train:
            raw_df.dropna(inplace=True)
            raw_df.reset_index(drop=True, inplace=True)
        return raw_df
    
    def clean_values(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        raw_df['country'] = raw_df['country'].replace({

        'UNITED ARAB EMIRATES':'UAE', # is same country
        'BURGARIA' : 'BULGARIA', # is same country
        
        'SCOTLAND': 'UNITED KINGDOM', # scotland it's in united kingdom 

        # Spelling mistakes    
        'SWIZERLAND': 'SWITZERLAND',
        'DJIBOUT': 'DJIBOUTI',
        'PHILIPINES': 'PHILIPPINES',
        'UKRAIN': 'UKRAINE',
        'SOMALI': 'SOMALIA',
        'MORROCO': 'MOROCCO',
        'SAUD ARABIA': 'SAUDI ARABIA',
        'COSTARICA': 'COSTA RICA',
        'COMORO': 'COMOROS',
        'ECUADO': 'ECUADOR',
        'MALT': 'MALTA',
        'TRINIDAD TOBACCO': 'TRINIDAD AND TOBAGO' # only in train data

        })
        raw_df['main_activity'] = raw_df['main_activity'].replace({
            'Widlife Tourism':'Wildlife Tourism'
        })
        return raw_df
    
    
    def process_pipeline(self, file_path: str) -> pd.DataFrame:
        load_file = self.load_data(file_path)
        check_dtypes = self.datatypes(load_file)
        clean_dupl = self.clean_duplicated(check_dtypes)
        clean_m_alone = self.missing_alone(clean_dupl)
        clean_m_spouse = self.missing_spouse(clean_m_alone)
        clean_missing = self.drop_missing(clean_m_spouse)
        clean_df = self.clean_values(clean_missing)
        process_df = clean_df
        return process_df

if __name__ == '__main__':
    train_df_pro = Data_Preprocessor(expected_columns=21, is_train=True)
    process_train_df = train_df_pro.process_pipeline(TRAIN_DATA_PATH)
    print(f"Train dataset was processed")
    print(f"missing value is {process_train_df.isna().sum().sum()} shape is {process_train_df.shape}")
   
    test_df_pro = Data_Preprocessor(expected_columns=20, is_train=False)
    process_test_df = test_df_pro.process_pipeline(TEST_DATA_PATH)
    print(f"Test dataset was processed")
    print(f"missing value is {process_test_df.isna().sum().sum()} shape is {process_test_df.shape}")
