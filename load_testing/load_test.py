import numpy as np
from locust import HttpUser, TaskSet, task, constant

# Defining a class to hit the server with GET and POST requests
class HitServer(TaskSet):
    @task(1) 
    def get_home(self):
        """Sending GET request to endpoint.."""
        self.client.get("/")
    
    @task(3) 
    def post_predict(self):
        """Sending request to predict endpoint.."""
        
        # Generate random data 
        data = {
            "buying_high": float(np.random.uniform(0, 1)),
            "buying_low": float(np.random.uniform(0, 1)),
            "buying_med": float(np.random.uniform(0, 1)),
            "buying_vhigh": float(np.random.uniform(0, 1)),
            "maint_high": float(np.random.uniform(0, 1)),
            "maint_low": float(np.random.uniform(0, 1)),
            "maint_med": float(np.random.uniform(0, 1)),
            "maint_vhigh": float(np.random.uniform(0, 1)),
            "doors_2": float(np.random.uniform(0, 1)),
            "doors_3": float(np.random.uniform(0, 1)),
            "doors_4": float(np.random.uniform(0, 1)),
            "doors_5more": float(np.random.uniform(0, 1)),
            "person_2": float(np.random.uniform(0, 1)),
            "person_4": float(np.random.uniform(0, 1)),
            "person_more": float(np.random.uniform(0, 1)),
            "lug_boot_big": float(np.random.uniform(0, 1)),
            "lug_boot_med": float(np.random.uniform(0, 1)),
            "lug_boot_small": float(np.random.uniform(0, 1)),
            "safety_high": float(np.random.uniform(0, 1)),
            "safety_low": float(np.random.uniform(0, 1)),
            "safety_med": float(np.random.uniform(0, 1)),
        }
        
        # Sending POST request to predict endpoint
        self.client.post(
            "/predict",
            json=data
        )

# Defining a class to load test the server
class UserLoadTest(HttpUser):
    host = "https://mlflow-staging.rcofwork.cloud"  # Define host URL
    tasks = [HitServer]                             # Define task to perform 
    wait_time = constant(1)                         # Define wait time between requests        