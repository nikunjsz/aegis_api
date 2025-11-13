# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import pandas as pd
# from pathlib import Path

# # IMPORTANT: import the class defined in api/crime_features.py
# from crime_features import CrimeFeatureEngineer


# app = FastAPI(title="Aegis Safety Score API")

# BASE_DIR = Path(__file__).resolve().parent
# MODEL_PATH = BASE_DIR / "safety_model_bundle.pkl"

# model = joblib.load(MODEL_PATH)

# class PredictRequest(BaseModel):
#     Community_Area: int
#     Month: int
#     Hour: int
#     Year: int

# @app.post("/predict")
# def predict(req: PredictRequest):
#     df = pd.DataFrame([{
#         "Community_Area": req.Community_Area,
#         "Month": req.Month,
#         "Hour": req.Hour,
#         "Year": req.Year
#     }])

#     y = model.predict(df)[0]
#     return {"severity_score": float(y)}
#  #uvicorn api.main:app --host 0.0.0.0 --port 10000

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# Import the feature engineering class if needed
from crime_features import CrimeFeatureEngineer

app = FastAPI(title="Aegis Safety Score API")

# ✅ Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict later e.g. ["https://yourflutterapp.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "safety_model_bundle.pkl"

model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    Community_Area: int
    Month: int
    Hour: int
    Year: int

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([{
        "Community_Area": req.Community_Area,
        "Month": req.Month,
        "Hour": req.Hour,
        "Year": req.Year
    }])

    y = model.predict(df)[0]
    return {"severity_score": float(y)}

# Command to run locally:
# uvicorn api.main:app --host 0.0.0.0 --port 10000
