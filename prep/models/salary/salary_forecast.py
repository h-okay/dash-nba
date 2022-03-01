# import pandas as pd
# import glob
# import os
# import xgboost as xgb
# import lightgbm as lgbm
# import catboost
# from sklearn.model_selection import (
#     train_test_split,
#     cross_validate,
#     cross_val_score,
#     RepeatedKFold,
# )
# from sklearn.preprocessing import RobustScaler, StandardScaler, OrdinalEncoder
# from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.model_selection import cross_validate
# import numpy as np
# from tqdm import tqdm
# import pickle as pkl
#
# from data.scripts.helpers import *

pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

salaries = pd.read_csv("data/base/salaries.csv")
salaries.YEAR.unique()
salaries.YEAR = salaries.YEAR.apply(lambda x: f"{x - 1}-{str(x)[2:]}")

all_teams = pd.read_csv("data/base/all_teams.csv").full_name.to_list()

all_players = pd.read_csv("data/base/all_players.csv")
all_players = all_players[all_players.is_active == True].full_name.to_list()

per = pd.read_csv("data/base/per.csv")
per["NAME"] = per["FIRST_NAME"] + " " + per["LAST_NAME"]

salaries = salaries[salaries.TEAM.isin(all_teams)].reset_index(drop=True)
temp = salaries.merge(
    per, left_on=["NAME", "YEAR"], right_on=["NAME", "SEASON_ID"], how="left"
)
temp.dropna(inplace=True)

temp.drop(
    [
        "FIRST_NAME",
        "LAST_NAME",
        "P_ID",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "factor",
        "vop",
        "drbp",
        "uPER",
        "T_PACE",
        "L_PACE",
        "adjustment",
        "aPER",
        "GS",
        "MIN",
        "FGM",
        "FGA",
        "FG3M",
        "FG3A",
        "FTM",
        "FTA",
        "TEAM_y",
        "YEAR",
    ],
    axis=1,
    inplace=True,
)

temp["REB"] = temp["REB"] / temp["GP"]
temp["AST"] = temp["AST"] / temp["GP"]
temp["STL"] = temp["STL"] / temp["GP"]
temp["BLK"] = temp["BLK"] / temp["GP"]
temp["TOV"] = temp["TOV"] / temp["GP"]
temp["PF"] = temp["PF"] / temp["GP"]
temp["PTS"] = temp["PTS"] / temp["GP"]
temp["SEASON"] = temp.SEASON_ID.apply(lambda x: int(x[:4]))
temp.drop("SEASON_ID", axis=1, inplace=True)
temp = temp.rename(columns={"TEAM_x": "TEAM"})

labels = temp[["NAME", "SALARY", "SEASON"]]
labels["SEASON"] = labels["SEASON"] - 1
labels = labels.rename(columns={"SALARY": "NEXT_SALARY"})
temp = temp.merge(labels, on=["SEASON", "NAME"], how="left").reset_index(
    drop=True)

current_season = temp[(temp.SEASON == 2021)]
current_season_check = current_season[["NAME", "TEAM", "SEASON", "SALARY"]]

temp.dropna(inplace=True)
temp.reset_index(drop=True, inplace=True)

check = temp[["NAME", "TEAM", "SEASON", "SALARY"]]
check = check[~check.duplicated()]

temp.columns
temp.drop(["SEASON", "NAME", "SALARY", "TEAM"], axis=1, inplace=True)
temp = temp[~temp.duplicated()]

results(temp, "NEXT_SALARY")

#                         CatB                      RF                      ET  \
# MSE  22316102531273.05859375 24125371900127.60156250 23951102192095.89843750
# R2                0.50000000              0.47000000              0.47000000
# RMSE        4582934.64000000        4745778.05000000        4728596.60000000
#                          XGB                    LGBM
# MSE  24194354989997.69921875 22061448572508.51953125
# R2                0.45000000              0.51000000
# RMSE        4791035.53000000        4557271.43000000


X = temp.drop("NEXT_SALARY", axis=1)
y = temp["NEXT_SALARY"]

cb = catboost.CatBoostRegressor(random_state=42, silent=True)
cb.fit(X, y)
y_pred = cb.predict(X)
temp["y_pred"] = y_pred
temp.y_pred = temp.y_pred.astype("int64")

# history
final = pd.concat([temp, check], axis=1)
final = final[["TEAM", "NAME", "SEASON", "SALARY", "y_pred"]]
final = final.rename(columns={"y_pred": "PRED"})
final.SALARY = final.SALARY.astype(np.compat.long)
final.SEASON = final.SEASON.astype("int64")
final.dropna(inplace=True)
final.head()
final.shape
