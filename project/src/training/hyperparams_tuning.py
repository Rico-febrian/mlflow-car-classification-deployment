import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
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

    # Set the hyperparameters grid
    params_grid = {
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [500, 1000],
        "C" : [0.1, 1, 10.0]
    }
    
    # Perform grid search for hyperparameters tuning
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42), 
        params_grid, 
        cv=3, 
        scoring="accuracy"
    )
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Set infer signature
    signature = infer_signature(X_train, y_train)
        
    # Log experiment to MLflow
    with mlflow.start_run(run_name="Hyperparameter Tuning - Second run") as run:
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", grid_search.best_score_)
        model_info = mlflow.sklearn.log_model(
            sk_model= best_model,
            artifact_path= "car_classification_models",
            signature=signature,
            input_example=X_train.loc[0].to_dict(),
            registered_model_name="Logistic Regression"
        )
        mlflow.log_param("status", "success")
    
    print(f"Best Model Parameters: {grid_search.best_params_}")    
    print(f"Model Accuracy: {grid_search.best_score_ * 100:.2f}%")
    print("Successfully saved the experiment run to MLflow!")