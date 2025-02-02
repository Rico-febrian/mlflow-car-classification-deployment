import os
import mlflow
import dotenv
import uvicorn
import numpy as np
import pandas as pd
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

# Define Pydantic model for input data
class api_data(BaseModel):
    buying_high: float
    buying_low: float
    buying_med: float
    buying_vhigh: float
    maint_high: float
    maint_low: float
    maint_med: float
    maint_vhigh: float
    doors_2: float
    doors_3: float
    doors_4: float
    doors_5more: float
    person_2: float
    person_4: float
    person_more: float
    lug_boot_big: float
    lug_boot_med: float
    lug_boot_small: float
    safety_high: float
    safety_low: float
    safety_med: float

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
        
        # Convert Pydantic model to dictionary
        input_dict = data.dict()
        
        # Convert dictionary to Pandas DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Make prediction
        y_pred = int(model.predict(input_df)[0])
        print("Prediction result:", y_pred)
        
        return {"res": y_pred, "error_msg": ""}
    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8081)