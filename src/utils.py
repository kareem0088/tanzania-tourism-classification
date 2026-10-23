from pathlib import Path
import sys 
import numpy as np
import pandas as pd

notebook_dir = Path.cwd()
project_root = notebook_dir.parent 

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import NUMERIC_COLS

#----------------------------------------------------------------------
#  outliers analysis (outliers count and percentage). 
#---------------------------------------------------------------------
def outliers(df):
    for col in NUMERIC_COLS:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1 
        lower_b = Q1 - 1.5 * IQR
        upper_b = Q3 + 1.5 * IQR

        outliers_count = df[(df[col] < lower_b) | (df[col] > upper_b)][col].count()
        count_per = outliers_count / len(df) * 100

        print(f"{col} | outliers : {outliers_count} - {count_per:.2f}%")

#----------------------------------------------------------------------
#  massing value analysis (number of missing values and percentage of missing values).
#---------------------------------------------------------------------
def massing_analysis(df):
    dic = {

    }

    for col in df.columns:
        nan = df[col].isna().sum()
        len_of_df = len(df[col])
        re = nan / len_of_df * 100
        dic[col] = {  
        'number of missing values': nan,
        'percentage of missing values': re.astype('float32').round(2)
        }

    df_missing = pd.DataFrame(dic).T
    df_missing = df_missing.replace(0, '')
    return df_missing