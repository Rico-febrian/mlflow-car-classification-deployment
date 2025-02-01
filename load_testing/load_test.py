import numpy as np

from random import randint
from locust import TaskSet, constant, task, HttpUser

class HitServer(TaskSet):
    @task
    def get_url(self):
        self.client.get("/")
        
class HitEndpoint(TaskSet):
    @task
    def post_predict(self):
        data = {
            "x1" : float(np.random.uniform(1, 10, 1)[0]),
            "x2" : float(np.random.uniform(1, 10, 1)[0]),
            "x3" : float(np.random.uniform(1, 10, 1)[0]),
            "x4" : float(np.random.uniform(1, 10, 1)[0])
        }
        
        self.client.post(
            "/predict",
            json = data
        )

class UserLoadTest(HttpUser):
    host = "https://mlflow-flower-clsf.rcofwork.cloud"
    tasks = [HitEndpoint]
    wait_time = constant(1)