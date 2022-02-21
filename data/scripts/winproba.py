import pandas as pd
import pickle as pkl
import numpy as np
from time import sleep
import re
from tqdm import tqdm


import sklearn.metrics
from sklearn.model_selection import train_test_split
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service

pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)

# Get Monthly Performances / Selenium
def get_monthly():
    all_data = pd.DataFrame()
    timeout = 8
    ser = Service("C:/Program Files/chromedriver.exe")
    s = webdriver.Chrome(service=ser)
    for id in all_teams.id.unique():
        if id == 1610612766:
            for season in edt.SEASON.unique()[1:]:
                random = np.random.random() * 3
                url = f"https://www.nba.com/stats/team/{id}/traditional/?Season={season}&SeasonType=Regular%20Season&PerMode=Totals"
                s.get(url)
                element_present = EC.presence_of_element_located(
                    (By.CLASS_NAME, "nba-stat-table__overflow")
                )
                WebDriverWait(s, timeout).until(element_present)
                html = s.page_source
                tables = pd.read_html(html)
                data = tables[6]
                data["TEAM"] = all_teams[all_teams.id == id].full_name.values[0]
                data["SEASON"] = season
                all_data = pd.concat([all_data, data], axis=0, ignore_index=True)
                sleep(0.6 + random)
        else:
            for season in edt.SEASON.unique():
                random = np.random.random() * 3
                url = f"https://www.nba.com/stats/team/{id}/traditional/?Season={season}&SeasonType=Regular%20Season&PerMode=Totals"
                s.get(url)
                element_present = EC.presence_of_element_located(
                    (By.CLASS_NAME, "nba-stat-table__overflow")
                )
                WebDriverWait(s, timeout).until(element_present)
                html = s.page_source
                tables = pd.read_html(html)
                data = tables[6]
                data["TEAM"] = all_teams[all_teams.id == id].full_name.values[0]
                data["SEASON"] = season
                all_data = pd.concat([all_data, data], axis=0, ignore_index=True)
                sleep(0.6 + random)

    s.quit()
    all_data.reset_index(drop=True, inplace=True)
    all_data.to_csv("monthly_team_perf.csv", index=False)


def get_matches():
    with open("data/base/matches.pkl", "rb") as file:
        matches = pkl.load(file)

    for i in matches.values():
        cols = i.columns
        break

    starter = pd.DataFrame(columns=cols)
    for value in tqdm(matches.values()):
        starter = pd.concat([starter, value]).copy()
    starter["AWAY"] = starter.apply(lambda x: "@" in x["MATCHUP"], axis=1) * 1
    away_matches = starter[starter["AWAY"] == 1].copy()
    home_matches = starter[starter["AWAY"] == 0].copy()
    home_matches["WL_away"] = home_matches["WL"].apply(
        lambda x: "W" if x == "L" else "L"
    )
    home_matches["ABB_away"] = home_matches["MATCHUP"].apply(lambda x: x[-3:])
    concat_matches = home_matches.merge(
        away_matches,
        left_on=["GAME_ID", "WL_away", "ABB_away"],
        right_on=["GAME_ID", "WL", "TEAM_ABBREVIATION"],
    )
    concat_matches = concat_matches[
        [
            "SEASON_ID_x",
            "GAME_DATE_x",
            "TEAM_ID_x",
            "TEAM_NAME_x",
            "TEAM_ID_y",
            "TEAM_NAME_y",
            "WL_x",
            "WL_y",
            "MIN_x",
            "PTS_x",
            "PTS_y",
        ]
    ]

    team_ids = concat_matches["TEAM_ID_x"].unique()
    a = concat_matches[["TEAM_ID_x", "GAME_DATE_x"]]
    checkpoint = concat_matches[~a.duplicated()]  # 6 maç drop oldu.
    checkpoint = (
        checkpoint.sort_values("GAME_DATE_x").reset_index().drop("index", axis=1)
    )
    checkpoint.SEASON_ID_x = checkpoint.SEASON_ID_x.apply(
        lambda x: "2009-10" if x == "2009-010" else x
    )
    checkpoint.GAME_DATE_x = pd.to_datetime(checkpoint.GAME_DATE_x)
    checkpoint["MONTH"] = checkpoint.GAME_DATE_x.dt.month_name()
    checkpoint.drop(["TEAM_ID_x", "TEAM_ID_y", "MIN_x"], axis=1, inplace=True)
    return checkpoint


# Data Import
all_teams = pd.read_csv("data/base/all_teams.csv")
all_teams = all_teams[["id", "full_name"]]
elo_ts = pd.read_csv("data/base/save_elo_ts.csv")
df = (
    elo_ts.merge(all_teams, left_on="TEAM_ID", right_on="id")
    .drop("id", axis=1)
    .rename(columns={"full_name": "TEAM"})
)
df = df.sort_values(by="DATE").reset_index(drop=True)
df.DATE = pd.to_datetime(df.DATE)
df["MONTH"] = df.DATE.dt.month_name()
df.head()
df.tail()

# Get monthly averages
agg_df = df.groupby(["SEASON", "TEAM", "MONTH"]).ELO.mean().reset_index()
agg_df[
    (agg_df.SEASON == "2021-22") & (agg_df.TEAM == "Toronto Raptors")
].MONTH.value_counts()  # 1 for each month as expected

monthly_elo = pd.DataFrame()  # 1 month shifted for ML purposes
for team in agg_df.TEAM.unique():
    temp = agg_df[(agg_df.TEAM == team)]
    temp.MONTH = temp.MONTH.shift(1)
    monthly_elo = pd.concat([monthly_elo, temp])

monthly_elo.isnull().sum()  # 30 as expected
monthly_elo.dropna(inplace=True)


monthly = pd.read_csv("data/base/monthly_team_perf.csv")
monthly = monthly.rename(columns={"Month": "MONTH"})
monthly.drop(["WIN%", "FG%", "3P%", "FT%", "+/-"], axis=1, inplace=True)
monthly.head()
monthly.tail()

monthlytotal = pd.DataFrame()  # Rolling totals for the month
for team in tqdm(monthly.TEAM.unique(), position=0, leave=True):
    for season in monthly.SEASON.unique():
        temp = monthly[(monthly.TEAM == team) & (monthly.SEASON == season)].reset_index(
            drop=True
        )
        for i, row in temp.iterrows():
            rolled = temp.rolling(i + 1).sum().dropna().iloc[0].to_frame().T
            rolled["MONTH"], rolled["SEASON"], rolled["TEAM"] = (
                temp.MONTH,
                temp.SEASON,
                temp.TEAM,
            )
            monthlytotal = pd.concat([monthlytotal, rolled])

lagged_stats = pd.DataFrame()  # 1 month shifted for ML purposes
for team in monthlytotal.TEAM.unique():
    temp = monthlytotal[(monthlytotal.TEAM == team)]
    temp.MONTH = temp.MONTH.shift(1)
    lagged_stats = pd.concat([lagged_stats, temp])

lagged_stats.isnull().sum()  # 30 as expected, first month stats needs to be dropped
lagged_stats.dropna(inplace=True)
lagged_stats.reset_index(drop=True, inplace=True)
lagged_stats.head()
lagged_stats.tail()

melo = monthly_elo.merge(lagged_stats, on=["SEASON", "MONTH", "TEAM"])

matches = get_matches()
matches.columns = [
    "SEASON_ID",
    "GAME_DATE",
    "TEAM1",
    "TEAM2",
    "WL1",
    "WL2",
    "PTS1",
    "PTS2",
    "MONTH",
]

home = matches[["SEASON_ID", "GAME_DATE", "TEAM1", "WL1", "PTS1", "MONTH"]]
home = home.merge(
    melo,
    left_on=["SEASON_ID", "MONTH", "TEAM1"],
    right_on=["SEASON", "MONTH", "TEAM"],
    how="left",
)
home.columns = [col + "1" for col in home.columns]
away = matches[["SEASON_ID", "GAME_DATE", "TEAM2", "WL2", "PTS2", "MONTH"]]
away = away.merge(
    melo,
    left_on=["SEASON_ID", "MONTH", "TEAM2"],
    right_on=["SEASON", "MONTH", "TEAM"],
    how="left",
)
away.columns = [col + "2" for col in away.columns]
final = pd.concat([home, away], axis=1)
final.dropna(inplace=True)
final.reset_index(drop=True)
final.shape  # (20837, 56)

final = final[
    [
        "SEASON1",
        "TEAM11",
        "WL11",
        "ELO1",
        "GP1",
        "MIN1",
        "PTS1",
        "W1",
        "L1",
        "FGM1",
        "FGA1",
        "3PM1",
        "3PA1",
        "FTM1",
        "FTA1",
        "OREB1",
        "DREB1",
        "REB1",
        "AST1",
        "TOV1",
        "STL1",
        "BLK1",
        "PF1",
        "TEAM22",
        "WL22",
        "ELO2",
        "GP2",
        "MIN2",
        "PTS2",
        "W2",
        "L2",
        "FGM2",
        "FGA2",
        "3PM2",
        "3PA2",
        "FTM2",
        "FTA2",
        "OREB2",
        "DREB2",
        "REB2",
        "AST2",
        "TOV2",
        "STL2",
        "BLK2",
        "PF2",
    ]
]


cols = (
    ["SEASON"]
    + [
        "HOME_" + re.findall("[a-zA-Z]+", col)[0]
        for col in final.columns
        if col[-1] == "1" and col != "SEASON1"
    ]
    + [
        "AWAY_" + re.findall("[a-zA-Z]+", col)[0]
        for col in final.columns
        if col[-1] == "2" and col != "SEASON2"
    ]
)
final.columns = cols
final = final.rename(
    columns={
        "HOME_PM": "HOME_3PM",
        "HOME_PA": "HOME_3PA",
        "AWAY_PM": "AWAY_3PM",
        "AWAY_PA": "AWAY_3PA",
    }
)

home_cols = [
    col
    for col in final.columns
    if "HOME" in col
    and col
    not in [
        "HOME_GP",
        "HOME_ELO",
        "HOME_W",
        "HOME_L",
        "HOME_WL",
        "HOME_TEAM",
        "HOME_MIN",
    ]
]
for col in home_cols:
    final[col] = final[col] / final["HOME_GP"]

away_cols = [
    col
    for col in final.columns
    if "AWAY" in col
    and col
    not in [
        "AWAY_GP",
        "AWAY_ELO",
        "AWAY_W",
        "AWAY_L",
        "AWAY_WL",
        "AWAY_TEAM",
        "AWAY_MIN",
    ]
]
for col in away_cols:
    final[col] = final[col] / final["AWAY_GP"]

final.drop(
    [
        "SEASON",
        "HOME_TEAM",
        "AWAY_TEAM",
        "HOME_MIN",
        "AWAY_MIN",
        "AWAY_WL",
        "HOME_PTS",
        "AWAY_PTS",
    ],
    axis=1,
    inplace=True,
)
final = pd.get_dummies(final, drop_first=True)
final = final.rename(columns={"HOME_WL_W": "OUTCOME"})  # 1 if home team wins
final.reset_index(drop=True, inplace=True)
final.head()


# Machine Learning
from data.scripts.myfunc import *

results(final, "OUTCOME")
#                F1 Score    Recall  Precision  Accuracy       ROC
# CatBoost       0.727655  0.803389   0.665582  0.643950  0.666470
# XGBoost        0.692242  0.737803   0.652194  0.611843  0.626507
# LigthGBM       0.725064  0.804428   0.660552  0.638767  0.662979
# LogisticReg    0.732914  0.819324   0.663571  0.646541  0.676168
# SVC            0.742838  0.914857   0.625983  0.624608  0.660044
# RandomForests  0.725019  0.813333   0.654712  0.634639  0.648416
# KNN            0.669598  0.706228   0.636704  0.587032  0.582167


import optuna
from sklearn.svm import SVC
from sklearn.metrics import f1_score

X = final.drop("OUTCOME", axis=1)
y = final["OUTCOME"]


def objective(trial):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    params = {
        "C": trial.suggest_float("C", 0.1, 1000),
        "kernel": trial.suggest_categorical(
            "kernel", ["linear", "poly", "rbf", "sigmoid"]
        ),
        "tol": trial.suggest_uniform("tol", 1e-5, 1e-1),
        "max_iter": trial.suggest_categorical("max_iter", [-1]),
    }

    if params["kernel"] in ["poly", "rbf", "sigmoid"]:
        params["gamma"] =  trial.suggest_categorical("gamma", ["scale", "auto"])
    if params["kernel"] in ["poly", "sigmoid"]:
        params["coef0"] = trial.suggest_discrete_uniform("coef0", 0.1, 1.9, 0.1)
    if params["kernel"] == 'poly':
        params['degree'] = trial.suggest_int('degree', 1, 100)

    model = SVC(**params, verbose=True).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred)
    return score

optuna.logging.set_verbosity(optuna.logging.INFO)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
