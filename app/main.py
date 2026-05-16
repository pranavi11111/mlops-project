from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Diabetes Prediction API", version="2.0.0")

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
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

class PredictionOutput(BaseModel):
    prediction: int
    confidence: float
    result: str

@app.get("/")
def home():
    return {
        "message": "Diabetes Prediction API is running! 🏥",
        "version": "2.0.0",
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
        input_data.mean_radius, input_data.mean_texture,
        input_data.mean_perimeter, input_data.mean_area,
        input_data.mean_smoothness, input_data.mean_compactness,
        input_data.mean_concavity, input_data.mean_concave_points,
        input_data.mean_symmetry, input_data.mean_fractal_dimension,
        input_data.radius_error, input_data.texture_error,
        input_data.perimeter_error, input_data.area_error,
        input_data.smoothness_error, input_data.compactness_error,
        input_data.concavity_error, input_data.concave_points_error,
        input_data.symmetry_error, input_data.fractal_dimension_error,
        input_data.worst_radius, input_data.worst_texture,
        input_data.worst_perimeter, input_data.worst_area,
        input_data.worst_smoothness, input_data.worst_compactness,
        input_data.worst_concavity, input_data.worst_concave_points,
        input_data.worst_symmetry, input_data.worst_fractal_dimension
    ]])

    prediction = model.predict(data)[0]
    confidence = float(max(model.predict_proba(data)[0]))
    result = "Malignant (Cancer)" if prediction == 0 else "Benign (No Cancer)"

    return PredictionOutput(
        prediction=int(prediction),
        confidence=round(confidence, 4),
        result=result
    )