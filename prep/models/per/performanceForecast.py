import pandas as pd
import glob
import os
import xgboost as xgb
import catboost
import lightgbm as lgbm
import catboost
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    cross_val_score,
    RepeatedKFold,
)
from sklearn.preprocessing import RobustScaler, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_validate
import numpy as np
from tqdm import tqdm
import pickle as pkl

from utils.helpers import *

pd.options.mode.chained_assignment = None


def get_player_perf_forecast():
    # Import and Process
    print("Import and process...")
    all_df = pd.read_csv("prep/data/per.csv")
    all_df.loc[all_df.LAST_NAME == "Yao Ming", "FIRST_NAME"] = "Yao"
    all_df.loc[all_df.LAST_NAME == "Yao Ming", "LAST_NAME"] = "Ming"
    all_df.loc[all_df.LAST_NAME == "Nene", "FIRST_NAME"] = "Nene"
    all_df.loc[all_df.LAST_NAME == "Nene", "LAST_NAME"] = "Nene"
    all_df.loc[all_df.LAST_NAME == "Yi Jianlian", "FIRST_NAME"] = "Yi"
    all_df.loc[all_df.LAST_NAME == "Yi Jianlian", "LAST_NAME"] = "Jianlian"
    all_df["NAME"] = all_df["FIRST_NAME"] + " " + all_df["LAST_NAME"]

    all_df = pd.read_csv("prep/data/per.csv")
    all_df["NAME"] = all_df["FIRST_NAME"] + " " + all_df["LAST_NAME"]
    all_df.drop(
        [
            "FIRST_NAME",
            "LAST_NAME",
            "P_ID",
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "factor",
            "vop",
            "drbp",
            "GS",
            "MIN",
            "FGM",
            "FGA",
            "FG3M",
            "FG3A",
            "FTM",
            "FTA",
        ],
        axis=1,
        inplace=True,
    )
    all_df
    all_df["REB"] = all_df["REB"] / all_df["GP"]
    all_df["AST"] = all_df["AST"] / all_df["GP"]
    all_df["STL"] = all_df["STL"] / all_df["GP"]
    all_df["BLK"] = all_df["BLK"] / all_df["GP"]
    all_df["TOV"] = all_df["TOV"] / all_df["GP"]
    all_df["PF"] = all_df["PF"] / all_df["GP"]
    all_df["PTS"] = all_df["PTS"] / all_df["GP"]
    all_df["SEASON"] = all_df.SEASON_ID.apply(lambda x: int(x[:4]))

    # Lagging
    labels = all_df[["NAME", "PER", "SEASON"]]
    labels["SEASON"] = labels["SEASON"] - 1
    labels = labels.rename(columns={"PER": "NEXT_PER"})
    all_df = all_df.merge(labels, on=["SEASON", "NAME"], how="left")

    current_season = all_df[(all_df.SEASON == 2021)]
    current_season_check = current_season[["NAME", "PER", "SEASON_ID", "TEAM"]]

    all_df.dropna(inplace=True)
    all_df.reset_index(drop=True, inplace=True)

    check = all_df[["NAME", "PER", "SEASON_ID", "TEAM"]]
    check = check[~check.duplicated()]  # to add later

    all_df.drop(["SEASON_ID", "SEASON", "NAME", "PER", "TEAM"], axis=1, inplace=True)
    all_df = all_df[~all_df.duplicated()]

    # Get Predictions
    print("Modelling...")
    X = all_df.drop("NEXT_PER", axis=1)
    y = all_df["NEXT_PER"]

    cb = catboost.CatBoostRegressor(random_state=42, silent=True)
    cb.fit(X, y)
    y_pred = cb.predict(X)
    all_df["y_pred"] = y_pred

    # history
    final = pd.concat([all_df, check], axis=1)
    final = final[["TEAM", "NAME", "SEASON_ID", "PER", "y_pred"]]
    final = final.rename(columns={"y_pred": "PRED"})

    # current
    current_season.drop(
        ["SEASON_ID", "SEASON", "NAME", "PER", "TEAM", "NEXT_PER"], axis=1, inplace=True
    )
    current_season["PRED"] = cb.predict(current_season)
    current_season = pd.concat([current_season, current_season_check], axis=1)
    current_season = current_season[["TEAM", "NAME", "SEASON_ID", "PER", "PRED"]]
    current_season.head()

    final = pd.concat([final, current_season], axis=0)
    final.to_csv("prep/estimations/perf_forecast.csv", index=False)
    print("Done.")
