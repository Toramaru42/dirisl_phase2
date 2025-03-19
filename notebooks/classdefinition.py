import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#データ加工クラスの定義
class Dataframeflag:
    def __init__(self, mappings):
        self.mappings = mappings

    def transform(self, df):
        for col, mapping in self.mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        return df

class DataframeLogScaler:
    def __init__(self, columns):
        self.columns = columns
        self.scaler = StandardScaler()
    def transform(self, df):
        df = df.copy()
        for col in self.columns:
            if col in df.columns:
                df[col] = np.log(df[col])
                df[col] = self.scaler.fit_transform(df[[col]])
        return df