from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="MLOps API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model Safely
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

# Input Schema
class PredictionInput(BaseModel):
    age: int
    income: int
    education: int
    experience: int

# Output Schema
class PredictionOutput(BaseModel):
    prediction: int
    confidence: float
    result: str

@app.get("/")
def home():
    return {
        "message": "MLOps API is running! 🚀",
        "version": "1.0.0",
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
    
    # Prepare input
    data = np.array([[input_data.age, input_data.income, 
                      input_data.education, input_data.experience]])
    
    # Predict
    prediction = model.predict(data)[0]
    confidence = float(max(model.predict_proba(data)[0]))
    result = "Survived" if prediction == 1 else "Not Survived"
    
    return PredictionOutput(
        prediction=int(prediction),
        confidence=round(confidence, 4),
        result=result
    )