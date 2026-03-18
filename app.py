from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

# ── Load model once at startup ─────────────────────────────
MODEL_PATH = "models/model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Run train.py first.")

model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")

# ── App setup ──────────────────────────────────────────────
app = FastAPI(
    title="Mental Health Text Detector",
    description="Detects mental health signals from text using NLP",
    version="1.0.0"
)

# Allow Streamlit (running on different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/response models ────────────────────────────────
class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float
    all_scores: dict

# ── Routes ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "API is running", "docs": "/docs"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    text = [input.text]
    prediction = model.predict(text)[0]
    probabilities = model.predict_proba(text)[0]
    labels = model.classes_

    scores = {
        label: round(float(prob), 4)
        for label, prob in zip(labels, probabilities)
    }

    return PredictionOutput(
        prediction=prediction,
        confidence=round(float(max(probabilities)), 4),
        all_scores=scores
    )