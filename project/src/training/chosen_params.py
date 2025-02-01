import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature
from dotenv import load_dotenv

# Load environment variables
load_dotenv("../../.env")

# Define the environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if __name__ == "__main__":
    
    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Car Classification")
    
    # Load and prepare data
    data = pd.read_pickle("../../data/processed/car_dataset.pkl")
    X = data.drop(columns=["target"])
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set the chosen parameters
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42,
    }
    
    # Train model with chosen parameters
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log experiment to MLflow
    with mlflow.start_run(run_name="Chosen Parameters") as run:
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        signature = infer_signature(X_train, y_train)
        mlflow.sklearn.log_model(model, "model", signature=signature)
        mlflow.log_param("status", "success")
        
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Successfully saved the experiment run to MLflow!")