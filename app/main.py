from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Diabetes Predictor API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

def load_model():
    global model
    model_path = "model/model.pkl"
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print(f"⚠️ Model not found at {model_path}. Please train the model first.")

load_model()

class PredictionInput(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: float

class PredictionOutput(BaseModel):
    prediction: int
    confidence: float
    result: str

@app.get("/")
def home():
    return {
        "message": "Diabetes Predictor API is running! 🏥",
        "version": "3.0.0",
        "status": "healthy"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )

    data = np.array([[
        input_data.pregnancies,
        input_data.glucose,
        input_data.blood_pressure,
        input_data.skin_thickness,
        input_data.insulin,
        input_data.bmi,
        input_data.diabetes_pedigree,
        input_data.age
    ]])

    prediction = model.predict(data)[0]
    confidence = float(max(model.predict_proba(data)[0]))
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"

    return PredictionOutput(
        prediction=int(prediction),
        confidence=round(confidence, 4),
        result=result
    )