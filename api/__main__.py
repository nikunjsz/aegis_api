# __main__.py — Fake module to satisfy pickle
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CrimeFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, history_map=None):
        self.history_map = history_map or {}

    def fit(self, X, y=None):
        # No fitting logic; preprocessing is deterministic
        return self

    def transform(self, X):
        df = X.copy()

        # deterministic features
        df["Quarter"] = ((df["Month"] - 1) // 3) + 1

        def get_season(m):
            if m in (12, 1, 2): return "winter"
            if m in (3, 4, 5): return "spring"
            if m in (6, 7, 8): return "summer"
            return "fall"

        df["Season"] = df["Month"].apply(get_season)

        def time_bucket(h):
            if 0 <= h < 6: return "late_night"
            if 6 <= h < 12: return "morning"
            if 12 <= h < 17: return "afternoon"
            if 17 <= h < 21: return "evening"
            return "night"

        df["Time_Bucket"] = df["Hour"].apply(time_bucket)

        # lag/roll values pulled from stored history_map
        df["Lag_1"] = df.apply(lambda r: self.history_map.get((r["Community_Area"], r["Year"], r["Month"], r["Hour"] - 1)), axis=1)
        df["Lag_2"] = df.apply(lambda r: self.history_map.get((r["Community_Area"], r["Year"], r["Month"], r["Hour"] - 2)), axis=1)
        df["Lag_3"] = df.apply(lambda r: self.history_map.get((r["Community_Area"], r["Year"], r["Month"], r["Hour"] - 3)), axis=1)

        for col in ["Lag_1", "Lag_2", "Lag_3"]:
            df[col] = df[col].fillna(0)

        return df
