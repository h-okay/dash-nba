# import catboost
import glob

# import lightgbm as lgbm
import numpy as np
import os
import pandas as pd

# import xgboost as xgb
# from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.model_selection import cross_validate
# from sklearn.model_selection import (
#     train_test_split,
#     cross_validate,
#     cross_val_score,
#     RepeatedKFold,
# )
# from sklearn.preprocessing import RobustScaler, StandardScaler, OrdinalEncoder
from tqdm import tqdm


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


# def validate(model, X, y):
#     results = pd.DataFrame(
#         cross_validate(model, X, y, cv=5,
#                        scoring=["neg_mean_squared_error", "r2"])
#     )
#     results["test_neg_mean_squared_error"] = results[
#         "test_neg_mean_squared_error"
#     ].apply(lambda x: -x)
#     results["rmse"] = results["test_neg_mean_squared_error"].apply(
#         lambda x: np.sqrt(x))
#     return results.mean().to_frame().T
#
#
# def results(dataframe, target, scale=False, ordinal=False):
#     X = dataframe.drop(target, axis=1)
#     y = dataframe[target]
#
#     if scale:
#         cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(X)
#         ss = StandardScaler()
#         for col in num_cols:
#             X[col] = ss.fit_transform(X[[col]])
#
#     if ordinal:
#         cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(X)
#         for col in cat_cols:
#             if X[col].dtype.name == "category":
#                 oe = OrdinalEncoder(
#                     categories=[X[col].dtype.categories.to_list()])
#                 X[col] = oe.fit_transform(X[[col]])
#
#     X = pd.get_dummies(X, drop_first=True)
#
#     models = [
#         catboost.CatBoostRegressor(random_state=42, silent=True),
#         RandomForestRegressor(random_state=42),
#         ExtraTreesRegressor(random_state=42),
#         xgb.XGBRegressor(random_state=42),
#         lgbm.LGBMRegressor(random_state=42),
#     ]
#
#     result = pd.DataFrame()
#     for model in tqdm(models, desc="Fitting "):
#         mdl = model
#         res = validate(mdl, X, y)
#         result = pd.concat([result, res])
#
#     result.index = ["CatB", "RF", "ET", "XGB", "LGBM"]
#     result = result[["test_neg_mean_squared_error", "test_r2", "rmse"]]
#     result = result.rename(
#         columns={"test_neg_mean_squared_error": "MSE", "test_r2": "R2",
#                  "rmse": "RMSE"}
#     )
#     return result.T


def fix_team_names(row):
    if row in ["New Jersey Nets", "New York Nets", "Brooklyn"]:
        return "Brooklyn Nets"
    if row in [
        "Washington Bullets",
        "Baltimore Bullets",
        "Capital Bullets",
        "Washington",
    ]:
        return "Washington Wizards"
    if row in ["LA Clippers", "San Diego Clippers", "Buffalo Braves", "L.A. Clippers"]:
        return "Los Angeles Clippers"
    if row in ["Kansas City Kings", "Cincinnati Royals", "Kansas City", "Sacramento"]:
        return "Sacramento Kings"
    if row in ["Seattle SuperSonics", "Oklahoma"]:
        return "Oklahoma City Thunder"
    if row in [
        "New Orleans/Oklahoma City Hornets",
        "New Orleans Hornets",
        "New Orleans",
        "NO/Oklahoma City  Hornets",
        "NO/Oklahoma City Hornets",
    ]:
        return "New Orleans Pelicans"
    if row in ["Charlotte Bobcats", "Charlotte"]:
        return "Charlotte Hornets"
    if row in ["Vancouver Grizzlies", "Memphis"]:
        return "Memphis Grizzlies"
    if row in ["San Francisco Warriors", "Golden St."]:
        return "Golden State Warriors"
    if row in ["San Diego Rockets", "Houston"]:
        return "Houston Rockets"
    if row in ["New Orleans Jazz", "Utah"]:
        return "Utah Jazz"
    if row == "Boston":
        return "Boston Celtics"
    if row == "Atlanta":
        return "Atlanta Hawks"
    if row == "Chicago":
        return "Chicago Bulls"
    if row == "Cleveland":
        return "Cleveland Cavaliers"
    if row == "Dallas":
        return "Dallas Mavericks"
    if row == "Denver":
        return "Denver Nuggets"
    if row == "Detroit":
        return "Detroit Pistons"
    if row == "Indiana":
        return "Indiana Pacers"
    if row == "L.A. Lakers":
        return "Los Angeles Lakers"
    if row == "Miami":
        return "Miami Heat"
    if row == "Milwaukee":
        return "Milwaukee Bucks"
    if row == "Minnesota":
        return "Minnesota Timberwolves"
    if row == "New York":
        return "New York Knicks"
    if row == "Orlando":
        return "Orlando Magic"
    if row == "Philadelphia":
        return "Philadelphia 76ers"
    if row == "Phoenix":
        return "Phoenix Suns"
    if row == "Portland":
        return "Portland Trail Blazers"
    if row == "Toronto":
        return "Toronto Raptors"
    if row == "San Antonio":
        return "San Antonio Spurs"

    return row


def get_names(row):
    for i, v in enumerate(row):
        if v == "-":
            return row[:i].strip()
    return row
