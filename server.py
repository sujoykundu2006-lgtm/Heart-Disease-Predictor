from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load("heart_disease_backend_model.joblib")
    yield

app = FastAPI(title="Heart Disease Predictor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/predict")
def predict(data: PatientData):
    try:
        input_df = pd.DataFrame([data.model_dump()])
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "risk": "High Risk" if probability >= 0.50 else "Low Risk",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
