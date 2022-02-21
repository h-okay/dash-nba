import catboost
import lightgbm as lgbm
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import (
    cross_validate,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tqdm import tqdm

kfold = KFold(n_splits=5)


def validate(model, X, y):
    result = (
        pd.DataFrame(
            cross_validate(
                model,
                X,
                y,
                cv=kfold,
                scoring=["f1", "recall", "precision", "accuracy", "roc_auc"],
            )
        )
        .mean()
        .to_frame()
        .iloc[2:, :]
        .rename(
            index={
                "test_f1": "F1 Score",
                "test_recall": "Recall",
                "test_precision": "Precision",
                "test_accuracy": "Accuracy",
                "test_roc_auc": "ROC",
            }
        )
    )
    return result.T[["F1 Score", "Recall", "Precision", "Accuracy", "ROC"]]


def results(dataframe, target):
    X = dataframe.drop(target, axis=1)
    y = dataframe[target]

    models = [
        ("Catboost", catboost.CatBoostClassifier(random_state=42, silent=True)),
        (
            "XGBoost",
            xgb.XGBClassifier(random_state=42, verbosity=0, use_label_encoder=False),
        ),
        ("LightGBM", lgbm.LGBMClassifier(random_state=42)),
        ("LogisticRegression", LogisticRegression(random_state=42, max_iter=100000)),
        ("SVC", SVC(random_state=42)),
        ("RandomForests", RandomForestClassifier(random_state=42)),
        ("KNN", KNeighborsClassifier()),
    ]

    cv_result = pd.DataFrame()
    for model in tqdm(models, position=0, leave=True, desc=" CV "):
        res = validate(model[1], X, y)
        cv_result = pd.concat([cv_result, res])

    cv_result.index = [
        "CatBoost",
        "XGBoost",
        "LigthGBM",
        "LogisticReg",
        "SVC",
        "RandomForests",
        "KNN",
    ]
    return cv_result
