# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "svm_full_microbiome.joblib"
SCALER_PATH = MODEL_DIR / "svm_full_microbiome_scaler.joblib"
GENERA_PATH = MODEL_DIR / "genera_order.joblib"

# ----------------------------
# Load artifacts
# ----------------------------
svm_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
genera_order = joblib.load(GENERA_PATH)

print("✅ Model loaded:", type(svm_model))
print("✅ Scaler loaded:", type(scaler))
print("✅ Genera count:", len(genera_order))

# ----------------------------
# FastAPI setup
# ----------------------------
app = FastAPI()


# Simple root route so visiting the URL shows a welcome message
@app.get("/")
def read_root():
    return {
        "message": "✅ SVM backend is live! Use POST /predict to send genera and read counts for prediction."
    }


# Allow CORS for your front-end
origins = ["*"]  # replace "*" with your front-end URL in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Request schema
# ----------------------------
class PredictionInput(BaseModel):
    reads: list[float]  # abundances
    genera: list[str]  # corresponding genera names


# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict(data: PredictionInput):
    # Create feature vector in correct order
    feature_vector = np.zeros(len(genera_order))
    for genus, read in zip(data.genera, data.reads):
        if genus in genera_order:
            idx = genera_order.index(genus)
            feature_vector[idx] = read

    # Scale features
    feature_vector_scaled = scaler.transform([feature_vector])

    # Predict
    pred_label = svm_model.predict(feature_vector_scaled)[0]
    pred_prob = svm_model.predict_proba(feature_vector_scaled)[0, 1]

    return {
        "prediction_label": int(pred_label),
        "prediction_probability": float(pred_prob)
    }
