# train_pipeline.py
import pandas as pd
import numpy as np
import os
# Adjust path so crime_features is importable if executing from project root
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "api"))

from crime_features import CrimeFeatureEngineer  # imports from api/crime_features.py if in PYTHONPATH
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import joblib




DATA_PATH = "Dataset/mapped_crimes_dataset2.csv"
TRAIN_YEAR_CUTOFF = 2023

df = pd.read_csv(DATA_PATH)
required = ['Community_Area','Month','Hour','Year','Severity_Score']
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}.")

if 'ID' not in df.columns:
    df = df.reset_index().rename(columns={'index':'ID'})

df['Month'] = df['Month'].astype(int)
df['Hour'] = df['Hour'].astype(int)
df['Year'] = df['Year'].astype(int)
df['Community_Area'] = df['Community_Area'].astype(int)
df['Severity_Score'] = pd.to_numeric(df['Severity_Score'], errors='coerce')
df = df.dropna(subset=['Severity_Score'])

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

df['Season'] = df['Month'].apply(get_season)
df['Quarter'] = ((df['Month'] - 1)//3) + 1
df['Time_Bucket'] = df['Hour'].apply(time_bucket)

agg_cols = ['Community_Area','Year','Month','Hour']
train_df = df[df['Year'] < TRAIN_YEAR_CUTOFF].copy()
test_df  = df[df['Year'] >= TRAIN_YEAR_CUTOFF].copy()

train_agg = train_df.groupby(agg_cols, as_index=False).agg({'Severity_Score':'mean','ID':'count'}).rename(columns={'ID':'Crime_Count'})
train_agg['Season'] = train_agg['Month'].apply(get_season)
train_agg['Quarter'] = ((train_agg['Month'] - 1)//3) + 1
train_agg['Time_Bucket'] = train_agg['Hour'].apply(time_bucket)
train_agg['ts'] = pd.to_datetime(dict(year=train_agg.Year, month=train_agg.Month, day=1)) + pd.to_timedelta(train_agg.Hour, unit='h')

# Prepare X_fit and y_fit for fe.fit: includes Severity_Score & Crime_Count & ts
X_fit = train_agg[['Community_Area','Month','Hour','Year','Season','Quarter','Time_Bucket','Crime_Count','Severity_Score','ts']].copy()
y_fit = train_agg['Severity_Score']

# Build pipeline
cat_cols = ['Community_Area','Season','Time_Bucket']
num_cols = ['Year','Month','Hour','Quarter','Crime_Count','Crime_Count_Ratio','Lag_1','Lag_2','Lag_3','Roll_3','Roll_6']

preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', 'passthrough', num_cols)
])

xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0, tree_method='hist')

pipe = Pipeline([
    ('fe', CrimeFeatureEngineer()),   # fit uses X_fit which contains history
    ('pre', preprocess),
    ('model', xgb)
])

# Fit pipeline (CrimeFeatureEngineer will build history_map using X_fit)
print("Fitting pipeline... (this may take a while)")
pipe.fit(X_fit, y_fit)

# Save pipeline into api folder (so API can load it)
out_path = os.path.join(os.path.dirname(__file__), "api", "safety_model_bundle.pkl")
joblib.dump(pipe, out_path)
print("Saved pipeline to:", out_path)
