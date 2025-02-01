import os
import mlflow
import dotenv
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load environment variables
dotenv.load_dotenv(".env")

# Set up mlflow tracking server and model
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
CHOSEN_MODEL = os.getenv("MODEL_ALIAS")

print("MLFLOW_TRACKING_URI:", MLFLOW_TRACKING_URI)
print("CHOSEN_MODEL:", CHOSEN_MODEL)

# Set mlflow url location
mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)

# Set mlflow experiment name
mlflow.set_experiment("Car Classification")

# Set up pyfunc to load model
print("Loading model...")
model = mlflow.pyfunc.load_model(f"models:/Logistic Regression@{CHOSEN_MODEL}")
print("Model loaded successfully.")

# Set up predictor
class api_data(BaseModel):
    x1: float
    x2: float
    x3: float
    x4: float

# Create FastAPI Object
app = FastAPI()

# Create home object
@app.get("/")
def home():
    return "Hello, FastAPI UP!"

# Create predict object
@app.post("/predict/")
def predict(data: api_data):
    try:
        print("Input data:", data)
        data = np.array([[data.x1, data.x2, data.x3, data.x4]]).astype(np.float64)
        y_pred = int(model.predict(data)[0])
        print("Prediction result:", y_pred)
        return {"res": y_pred, "error_msg": ""}
    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080)