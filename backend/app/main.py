from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
import joblib

app = FastAPI(title="SMS Spam Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

class SpamRequest(BaseModel):
    message: str

class SpamResponse(BaseModel):
    prediction: str
    confidence: float

@app.get("/")
def home():
    return {"message": "SMS Spam Classifier API is running"}

@app.get("/docs")
def docs():
    return {"message": "Open Swagger docs at /docs"}

@app.post("/predict", response_model=SpamResponse)
def predict_spam(req: SpamRequest):
    try:
        prediction = model.predict([req.message])[0]
        probabilities = model.predict_proba([req.message])[0]
        
        if prediction == "spam":
            confidence = probabilities[1]
        else:
            confidence = probabilities[0]
        
        return SpamResponse(
            prediction=prediction,
            confidence=round(confidence, 3)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.get("/test")
def test_endpoint():
    test_messages = [
        "Congratulations! You won a free iPhone. Call now!",
        "Hey, are you coming to class today?",
        "Urgent! Claim your prize money now.",
        "Can you send me the notes after lunch?"
    ]
    
    results = []
    for msg in test_messages:
        prediction = model.predict([msg])[0]
        probabilities = model.predict_proba([msg])[0]
        confidence = probabilities[1] if prediction == "spam" else probabilities[0]
        
        results.append({
            "message": msg,
            "prediction": prediction,
            "confidence": round(confidence, 3)
        })
    
    return results