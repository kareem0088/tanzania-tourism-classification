"""
src/features/encoding.py
==========================
Module Specifications:
- Process and encode categorical features and the target variable.
- This file is called exclusively after the train-test split process to prevent data leakage.
"""

import pandas as pd
from src.config import CATEGORICAL_COLS
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

class Encoder:
    def __init__(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    def encode_categorical(self) -> pd.DataFrame:
        encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
        )

        self.X_train[CATEGORICAL_COLS] = encoder.fit_transform(self.X_train)
        self.X_test[CATEGORICAL_COLS] = encoder.transform(self.X_test)

        label_encoder = LabelEncoder()

        self.y_train = label_encoder.fit_transform(self.y_train)
        self.y_test = label_encoder.transform(self.y_test)
        return self.X_train, self.X_test, self.y_train, self.y_test

''' Note: We decided to keep the standard encoding for
    the country column since we are using tree-based models,
    which handle ordinal-encoded categorical features 
    effectively without dimensionality explosion. '''