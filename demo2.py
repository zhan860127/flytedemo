import json
import os
import joblib
import pandas as pd

from flytekit import task, workflow, Resources , ImageSpec
from flytekit.types.directory import FlyteDirectory
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


training_image = ImageSpec(
    name="real-cpu-train-2",
    packages=[
        "torch", 
        "flytekit", 
        "cachetools", # 強制加入以解決 ModuleNotFoundError
        "pandas",
        "scikit-learn"
    ],
    registry="docker.io/zhan860127" 
)

@task(requests=Resources(mem="700Mi"),container_image=training_image,)
def get_data() -> pd.DataFrame:
    """Load the wine dataset as a DataFrame."""
    return load_wine(as_frame=True).frame

@task
def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the original 3-class problem into binary classification:
    class 0 -> 0
    class 1,2 -> 1
    """
    return data.assign(
        target=lambda df: df["target"].where(df["target"] == 0, 1)
    )


@task(cache=True, cache_version="train-v1")
def train_model(
    data: pd.DataFrame,
    hyperparameters_json: str,
) -> FlyteDirectory:
    """
    Train a Logistic Regression model and persist it as an artifact.
    """
    hyperparameters = json.loads(hyperparameters_json)

    X = data.drop("target", axis="columns")
    y = data["target"]

    model = LogisticRegression(**hyperparameters)
    model.fit(X, y)

    output_dir = "/tmp/model"
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_path)

    return FlyteDirectory(path=output_dir)

@task
def evaluate_model(
    model_dir: FlyteDirectory,
    data: pd.DataFrame,
) -> float:
    """
    Load the trained model and evaluate accuracy.
    """
    model_dir.download()
    model_path = os.path.join(model_dir.path, "model.joblib")

    model = joblib.load(model_path)

    X = data.drop("target", axis="columns")
    y = data["target"]

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    print(f"Model accuracy: {accuracy:.4f}")
    return accuracy

@workflow
def training_workflow(
    hyperparameters_json: str = '{"C": 1.0, "max_iter": 200}'
) -> float:
    """
    End-to-end training workflow.
    """
    raw_data = get_data()
    processed_data = process_data(data=raw_data)

    model_dir = train_model(
        data=processed_data,
        hyperparameters_json=hyperparameters_json,
    )

    accuracy = evaluate_model(
        model_dir=model_dir,
        data=processed_data,
    )

    return accuracy

