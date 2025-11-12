# api/crime_features.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def get_season(month):
    m = int(month)
    if m in (12,1,2): return 'winter'
    if m in (3,4,5): return 'spring'
    if m in (6,7,8): return 'summer'
    return 'fall'

def time_bucket(hour):
    h = int(hour)
    if 0 <= h < 6: return 'late_night'
    if 6 <= h < 12: return 'morning'
    if 12 <= h < 17: return 'afternoon'
    if 17 <= h < 21: return 'evening'
    return 'night'


class CrimeFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.history_map = {}
        self.mean_crime_count = {}

    def fit(self, X, y=None):
        df_hist = X.copy()
        if 'ts' not in df_hist.columns:
            df_hist['ts'] = pd.to_datetime(dict(year=df_hist.Year, month=df_hist.Month, day=1)) + pd.to_timedelta(df_hist.Hour, unit='h')
        # build history_map and mean_crime_count
        for ca, g in df_hist.groupby('Community_Area'):
            g_sorted = g.sort_values('ts')
            self.history_map[int(ca)] = list(zip(g_sorted['ts'].tolist(), g_sorted['Severity_Score'].tolist()))
            self.mean_crime_count[int(ca)] = float(g_sorted['Crime_Count'].mean()) if 'Crime_Count' in g_sorted else 0.0
        return self

    def _compute_from_history(self, ca, ts):
        if int(ca) not in self.history_map:
            return {k: 0.0 for k in ['Lag_1','Lag_2','Lag_3','Roll_3','Roll_6','Crime_Count_Ratio']}
        series = self.history_map[int(ca)]
        past = [sev for (t, sev) in series if t < ts]
        lag1 = past[-1] if len(past) >= 1 else 0.0
        lag2 = past[-2] if len(past) >= 2 else 0.0
        lag3 = past[-3] if len(past) >= 3 else 0.0
        roll3 = float(np.mean(past[-3:])) if past else 0.0
        roll6 = float(np.mean(past[-6:])) if past else 0.0
        crime_mean = self.mean_crime_count.get(int(ca), 0.0)
        crime_ratio = 1.0 / (crime_mean + 1e-9) if crime_mean > 0 else 0.0
        return {'Lag_1':lag1,'Lag_2':lag2,'Lag_3':lag3,'Roll_3':roll3,'Roll_6':roll6,'Crime_Count_Ratio':crime_ratio}

    def transform(self, X):
        df = X.copy().reset_index(drop=True)
        df['Season'] = df['Month'].apply(get_season)
        df['Quarter'] = ((df['Month'] - 1) // 3) + 1
        df['Time_Bucket'] = df['Hour'].apply(time_bucket)
        df['Crime_Count'] = 1.0
        df['ts'] = pd.to_datetime(dict(year=df.Year, month=df.Month, day=1)) + pd.to_timedelta(df.Hour, unit='h')
        rows = []
        for _, row in df.iterrows():
            rows.append(self._compute_from_history(int(row['Community_Area']), row['ts']))
        lag_df = pd.DataFrame(rows)
        df = pd.concat([df, lag_df], axis=1)
        final_cols = ['Community_Area','Season','Time_Bucket','Year','Month','Hour','Quarter','Crime_Count','Crime_Count_Ratio','Lag_1','Lag_2','Lag_3','Roll_3','Roll_6']
        # ensure columns exist
        for col in final_cols:
            if col not in df.columns:
                df[col] = 0.0
        return df[final_cols]
