import typing
from dataclasses import dataclass
from typing import Tuple
import joblib
import pandas as pd
from dataclasses_json import dataclass_json
from flytekit import Resources, task, workflow ,ImageSpec,conditional
from flytekit.types.file import FlyteFile
from flytekit.types.structured import StructuredDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import json


training_image = ImageSpec(
    name="real-cpu-train-3",
    packages=[
        "torch", 
        "flytekit", 
        "cachetools", # 強制加入以解決 ModuleNotFoundError
        "pandas",
        "xgboost",
        "pyarrow",
        "scikit-learn"
    ],
    registry="docker.io/zhan860127" 
)
# -----------------------------
# 1️⃣ Hyperparameters Dataclass
# -----------------------------
@dataclass_json
@dataclass
class XGBoostModelHyperparams:
    max_depth: int = 3
    learning_rate: float = 0.1
    n_estimators: int = 100
    objective: str = "binary:logistic"
    booster: str = "gbtree"
    n_jobs: int = 1

FEATURE_COLS = [
    "#preg",
    "pgc_2h",
    "diastolic_bp",
    "tricep_skin_fold_mm",
    "serum_insulin_2h",
    "bmi",
    "diabetes_pedigree",
    "age",
]
LABEL_COLS = ["class"]

# -----------------------------
# 2️⃣ Tasks
# -----------------------------
@task(cache_version="1.2", cache=True, limits=Resources(mem="500Mi", cpu="2") ,container_image=training_image)
def split_traintest_dataset(
    dataset: FlyteFile, seed: int, test_split_ratio: float
) -> Tuple[StructuredDataset, StructuredDataset, StructuredDataset, StructuredDataset]:
    column_names = FEATURE_COLS + LABEL_COLS
    df = pd.read_csv(dataset, names=column_names)
    x = df[FEATURE_COLS]
    y = df[LABEL_COLS]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split_ratio, random_state=seed)
    return StructuredDataset(dataframe=x_train), StructuredDataset(dataframe=x_test), StructuredDataset(dataframe=y_train), StructuredDataset(dataframe=y_test)

MODELSER_JOBLIB = typing.TypeVar("joblib.dat")
ModelFile = typing.NamedTuple("ModelFile", model=FlyteFile[MODELSER_JOBLIB])
WorkflowOutputs = typing.NamedTuple("WorkflowOutputs", model=FlyteFile[MODELSER_JOBLIB], accuracy=float)

@task(cache_version="1.3", cache=False, limits=Resources(mem="1Gi", cpu="2"),container_image=training_image)
def fit(
    x: StructuredDataset,
    y: StructuredDataset,
    hyperparams_json: str,   # ✅ 這裡改成 JSON 字串
) -> ModelFile:
    hyperparams = XGBoostModelHyperparams.from_json(hyperparams_json)  # ✅ 轉回 dataclass
    x_df = x.open(pd.DataFrame).all()
    y_df = y.open(pd.DataFrame).all()

    model = XGBClassifier(
        n_jobs=hyperparams.n_jobs,
        max_depth=hyperparams.max_depth,
        n_estimators=hyperparams.n_estimators,
        booster=hyperparams.booster,
        objective=hyperparams.objective,
        learning_rate=hyperparams.learning_rate,
    )
    model.fit(x_df, y_df)

    fname = "/tmp/model.joblib.dat"
    joblib.dump(model, fname)
    return (FlyteFile(path=fname),)

@task(cache_version="1.2", cache=True, limits=Resources(mem="500Mi", cpu="2"),container_image=training_image)
def predict(x: StructuredDataset, model_ser: FlyteFile[MODELSER_JOBLIB]) -> StructuredDataset:
    model = joblib.load(model_ser)
    x_df = x.open(pd.DataFrame).all()
    y_pred = model.predict(x_df)
    return StructuredDataset(dataframe=pd.DataFrame(y_pred, columns=LABEL_COLS, dtype="int64"))

@task(cache_version="1.2", cache=True, limits=Resources(mem="500Mi", cpu="2"),container_image=training_image)
def score(predictions: StructuredDataset, y: StructuredDataset) -> float:
    pred_df = predictions.open(pd.DataFrame).all()
    y_df = y.open(pd.DataFrame).all()
    return float(accuracy_score(y_df, pred_df))

# -----------------------------
# 3️⃣ Remote-safe Workflow
# -----------------------------
@workflow
def diabetes_xgboost_model(
    dataset: FlyteFile = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    test_split_ratio: float = 0.33,
    seed: int = 7,
    hyperparams_json: str = XGBoostModelHyperparams(max_depth=4).to_json(),  # ✅ JSON 字串
) -> WorkflowOutputs:
    x_train, x_test, y_train, y_test = split_traintest_dataset(dataset=dataset, seed=seed, test_split_ratio=test_split_ratio)
    model = fit(x=x_train, y=y_train, hyperparams_json=hyperparams_json)
    predictions = predict(x=x_test, model_ser=model.model)
    return model.model, score(predictions=predictions, y=y_test)



