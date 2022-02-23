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
                scoring=["neg_log_loss", "f1", "roc_auc"],
            )
        )
        .mean()
        .to_frame()
        .iloc[2:, :]
        .rename(
            index={
                "test_f1": "F1 Score",
                "test_roc_auc": "ROC",
                "test_neg_log_loss": "Log-Loss",
            }
        )
    )
    results["Log-Loss"] = results["Log-Loss"].apply(lambda x: -x)
    return result.T[["Log-Loss", "F1 Score", "ROC"]]


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
    return cv_result.sort_values(
        by=["Log-Loss", "F1 Score", "ROC"], ascending=[True, False, False]
    )


def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"
    ]

    cat_but_car = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"
    ]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car, num_but_cat
