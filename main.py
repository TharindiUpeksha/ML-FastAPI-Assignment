# Load trained model and class names
model = joblib.load("model.pkl")
class_names = joblib.load("classes.pkl")
# Load trained model
model = joblib.load("model.pkl")

# Hardcode class names
class_names = ['setosa', 'versicolor', 'virginica']

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load trained model and class names
# Make sure model.pkl and classes.pkl are in the same folder
model = joblib.load("model.pkl")
class_names = joblib.load("classes.pkl")

# Create FastAPI app
app = FastAPI(
  title="Iris Classification API",
    description="Predict iris species (setosa, versicolor, virginica)",
    version="1.0"
)

# Input schema
class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Output schema
class PredictionOutput(BaseModel):
    prediction: str
    confidence: float

# Health check endpoint
@app.get("/")
def health_check():
  return {"status": "healthy", "message": "ML Model API is running"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        features = np.array([[input_data.sepal_length,
                              input_data.sepal_width,
                              input_data.petal_length,
                              input_data.petal_width]])
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features).max()

        return PredictionOutput(
            prediction=class_names[prediction],
            confidence=float(proba)
)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Model info endpoint
@app.get("/model-info")
def model_info():
    return {
        "model_type": "RandomForestClassifier",
        "problem_type": "classification",
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "classes": list(class_names)
    }
