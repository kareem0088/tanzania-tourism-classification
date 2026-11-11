import numpy as np
import pandas as pd

from src.config import TEST_DATA_PATH, TRAIN_DATA_PATH, FEATURES_DIR

continent_mapping = {
    'ITALY': 'EUROPE', 'UNITED KINGDOM': 'EUROPE', 'UNITED STATES OF AMERICA': 'NORTH AMERICA',
    'RWANDA': 'AFRICA', 'FRANCE': 'EUROPE', 'SWITZERLAND': 'EUROPE', 'NIGERIA': 'AFRICA',
    'AUSTRALIA': 'OCEANIA', 'SPAIN': 'EUROPE', 'DENMARK': 'EUROPE', 'BELGIUM': 'EUROPE',
    'ZAMBIA': 'AFRICA', 'NETHERLANDS': 'EUROPE', 'CANADA': 'NORTH AMERICA', 'INDIA': 'ASIA',
    'NORWAY': 'EUROPE', 'ZIMBABWE': 'AFRICA', 'MOROCCO': 'AFRICA', 'CHINA': 'ASIA',
    'GERMANY': 'EUROPE', 'SOUTH AFRICA': 'AFRICA', 'JAPAN': 'ASIA', 'UAE': 'ASIA',
    'RUSSIA': 'EUROPE/ASIA', 'KENYA': 'AFRICA', 'GREECE': 'EUROPE', 'POLAND': 'EUROPE',
    'AUSTRIA': 'EUROPE', 'MALAYSIA': 'ASIA', 'CZECH REPUBLIC': 'EUROPE', 'KOREA': 'ASIA',
    'DRC': 'AFRICA', 'CYPRUS': 'EUROPE', 'KUWAIT': 'ASIA', 'BRAZIL': 'SOUTH AMERICA',
    'CHILE': 'SOUTH AMERICA', 'BULGARIA': 'EUROPE', 'ISRAEL': 'ASIA', 'NEW ZEALAND': 'OCEANIA',
    'UGANDA': 'AFRICA', 'IRELAND': 'EUROPE', 'SWEDEN': 'EUROPE', 'QATAR': 'ASIA',
    'CONGO': 'AFRICA', 'BURUNDI': 'AFRICA', 'FINLAND': 'EUROPE', 'GHANA': 'AFRICA',
    'OMAN': 'ASIA', 'LEBANON': 'ASIA', 'LUXEMBOURG': 'EUROPE', 'IVORY COAST': 'AFRICA',
    'NAMIBIA': 'AFRICA', 'ROMANIA': 'EUROPE', 'THAILAND': 'ASIA', 'SUDAN': 'AFRICA',
    'SINGAPORE': 'ASIA', 'PORTUGAL': 'EUROPE', 'MALAWI': 'AFRICA', 'MALTA': 'EUROPE',
    'EGYPT': 'AFRICA', 'PAKISTAN': 'ASIA', 'TANZANIA': 'AFRICA', 'ARGENTINA': 'SOUTH AMERICA',
    'SRI LANKA': 'ASIA', 'BOTSWANA': 'AFRICA', 'VIETNAM': 'ASIA', 'BERMUDA': 'NORTH AMERICA',
    'TAIWAN': 'ASIA', 'SENEGAL': 'AFRICA', 'AFGHANISTAN': 'ASIA', 'COLOMBIA': 'SOUTH AMERICA',
    'JAMAICA': 'NORTH AMERICA', 'BAHRAIN': 'ASIA', 'COSTA RICA': 'NORTH AMERICA', 'COMOROS': 'AFRICA',
    'URUGUAY': 'SOUTH AMERICA', 'SAUDI ARABIA': 'ASIA', 'ETHIOPIA': 'AFRICA', 'PHILIPPINES': 'ASIA',
    'ARMENIA': 'ASIA', 'MEXICO': 'NORTH AMERICA', 'MADAGASCAR': 'AFRICA', 'SERBIA': 'EUROPE',
    'SLOVAKIA': 'EUROPE', 'IRAN': 'ASIA', 'SWAZILAND': 'AFRICA', 'PERU': 'SOUTH AMERICA',
    'LITHUANIA': 'EUROPE', 'ALGERIA': 'AFRICA', 'SEYCHELLES': 'AFRICA', 'MOZAMBIQUE': 'AFRICA',
    'UKRAINE': 'EUROPE', 'NEPAL': 'ASIA', 'CROATIA': 'EUROPE', 'SOMALIA': 'AFRICA',
    'LIBERIA': 'AFRICA', 'MAURITIUS': 'AFRICA', 'GAMBIA': 'AFRICA', 'ECUADOR': 'SOUTH AMERICA',
    'LATVIA': 'EUROPE', 'MACEDONIA': 'EUROPE', 'LESOTHO': 'AFRICA', 'HUNGARY': 'EUROPE',
    'ICELAND': 'EUROPE', 'VENEZUELA': 'SOUTH AMERICA', 'ANGOLA': 'AFRICA', 'SLOVENIA': 'EUROPE',
    'CAPE VERDE': 'AFRICA', 'CAMEROON': 'AFRICA', 'MONECASQUE': 'EUROPE', 'TRINIDAD AND TOBAGO': 'SOUTH AMERICA',
    'PAPUA NEW GUINEA': 'OCEANIA', 'TURKEY': 'EUROPE/ASIA', 'INDONESIA': 'ASIA', 'ESTONIA': 'EUROPE',
    'BOSNIA': 'EUROPE', 'YEMEN': 'ASIA', 'NIGER': 'AFRICA', 'ERITREA': 'AFRICA',
    'GEORGIA': 'ASIA', 'BANGLADESH': 'ASIA', 'BARBADOS': 'NORTH AMERICA', 'TUNISIA': 'AFRICA',
    'DJIBOUTI': 'AFRICA', 'DOMINICA': 'NORTH AMERICA', 'CAMBODIA': 'ASIA', 'MONTENEGRO': 'EUROPE',
    'JORDAN': 'ASIA'
}

class FeatureEngineering:
    def __init__(self, df_path: str):
        self.df = pd.read_csv(df_path)

    def create_features(self) -> pd.DataFrame:
        df = self.df.copy()
        df['total nights'] = df.night_mainland + df.night_zanzibar
        df['total_adults'] = df.total_female + df.total_male
        df['night_ratio'] = np.where(
            df['night_zanzibar'] == 0, 0,
            df['night_mainland'] / df['night_zanzibar']
        )
        df['adults_per_night'] = df['total_adults'] / df['total nights']
        df['adults_per_night'] = df['adults_per_night'].round(2)
        df['male_to_female_ratio'] = np.where(
            df['total_female'] == 0, 0,
            df['total_male'] / df['total_female']
        )
        df['male_to_female_ratio'] = df['male_to_female_ratio'].round(2)
        df['continent'] = df['country'].map(continent_mapping)
        df.reset_index(drop=True, inplace=True)

        return df

if __name__ == "__main__":
        processed_train_df = FeatureEngineering(TRAIN_DATA_PATH).create_features()
        processed_train_df.to_parquet(FEATURES_DIR / f"train_data_FE.parquet", index=False)
        
        processed_test_df = FeatureEngineering(TEST_DATA_PATH).create_features()
        processed_test_df.to_parquet(FEATURES_DIR / f"test_data_FE.parquet", index=False)
          
