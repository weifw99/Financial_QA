import os
from random import random, randint

import mlflow

if __name__ == "__main__":
    mlflow.set_tracking_uri('http://127.0.0.1:5001')

    experiments = mlflow.search_experiments(filter_string="name = 'test'")
    if len(experiments) != 1:
        experiment_id = mlflow.create_experiment('test')
        experiments = [experiment_id]
        experiment_id = experiments[0]
    else:
        experiment_id = experiments[0].experiment_id
    print(f"Experiment ID: {experiments}")

    with mlflow.start_run(experiment_id=experiment_id) as run:
        print("Running mlflow_tracking.py")

        mlflow.log_param("param1", randint(0, 100))

        mlflow.log_metric("foo", random())
        mlflow.log_metric("foo", random() + 1)
        mlflow.log_metric("foo", random() + 2)

        if not os.path.exists("outputs"):
            os.makedirs("outputs")
        with open("outputs/test.txt", "w") as f:
            f.write("hello world!")

        mlflow.log_artifacts("outputs")
