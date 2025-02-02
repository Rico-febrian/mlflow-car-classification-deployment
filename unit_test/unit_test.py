import os
import mlflow
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")

def test_model_availability():
    
    # Arrange
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    CHOSEN_MODEL = os.getenv("MODEL_ALIAS")
    
    mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Car Classification")
    
    # Act
    chosen_model = mlflow.pyfunc.load_model(f"models:/Logistic Regression@{CHOSEN_MODEL}")
    
    # Assert
    assert type(chosen_model) == mlflow.pyfunc.PyFuncModel


def test_predict_on_ci():
    
    # Arrange
    MODEL_SERVER_IP = os.getenv("MODEL_SERVER_IP")
    
    # Define input data
    data = {
        "buying_high": 1.0,
        "buying_low": 0.0,
        "buying_med": 0.0,
        "buying_vhigh": 0.0,
        "maint_high": 0.0,
        "maint_low": 1.0,
        "maint_med": 0.0,
        "maint_vhigh": 0.0,
        "doors_2": 1.0,
        "doors_3": 0.0,
        "doors_4": 0.0,
        "doors_5more": 0.0,
        "person_2": 0.0,
        "person_4": 1.0,
        "person_more": 0.0,
        "lug_boot_big": 0.0,
        "lug_boot_med": 1.0,
        "lug_boot_small": 0.0,
        "safety_high": 0.0,
        "safety_low": 1.0,
        "safety_med": 0.0
    }
    
    # Act
    try:
        res = requests.post(MODEL_SERVER_IP, json=data)
        print("Response Status Code:", res.status_code)  # Debug line
        print("Response Body:", res.json())  # Debug line
    except Exception as e:
        assert False, f"Failed to make request: {e}"
    
    # Assert
    assert res.status_code == 200