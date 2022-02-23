import pandas as pd
import pickle as pkl
import numpy as np
from time import sleep
import re
from tqdm import tqdm
import warnings
import lightgbm as lgbm

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
warnings.filterwarnings("ignore")

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

monthly_elo = pd.DataFrame()  # 1 month shifted for ML purposes / prevent data leakeage
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

lagged_stats = pd.DataFrame()  # 1 month shifted for ML purposes / prevent data leakeage
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
final.reset_index(drop=True, inplace=True)
final.shape  # (20837, 56)

final = final[
    [
        "SEASON1",
        "GAME_DATE1",
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

final.head()


# Adding rankings before match as features
# import re
# def daily_rankings(date):
#     year = date.split('-')[0]
#     month = date.split('-')[1]
#     day = date.split('-')[2]
#     url = f'https://www.basketball-reference.com/friv/standings.fcgi?month={month}&day={day}&year={year}&lg_id=NBA'
#     # Atlantic-Central
#     ac = pd.read_html(url)[0]
#     # ac.drop([0, 8], axis=0, inplace=True)
#     ac.drop('GB', axis=1, inplace=True)
#     ac.columns = ['TEAM', 'W', 'L', 'RANK', 'PW', 'PL', 'PS/G', 'PA/G']
#     #Midwest-Pacific
#     mp = pd.read_html(url)[1]
#     # mp.drop([0, 8], axis=0, inplace=True)
#     mp.drop('GB', axis=1, inplace=True)
#     mp.columns = ['TEAM', 'W', 'L', 'RANK', 'PW', 'PL', 'PS/G', 'PA/G']
#     acmp = pd.concat([ac, mp], axis=0)
#     if len(str(day)) == 1:
#         acmp['DATE'] = str(year) + "-" + str(month) + "-0" + str(day)
#     else:
#         acmp['DATE'] = str(year) + "-" + str(month) + "-" + str(day)
#     return acmp
#
# dates = final.GAME_DATE1.dt.date.apply(lambda x: x.strftime("%Y-%m-%d")).unique()
#
# rankings = pd.DataFrame()
# for date in tqdm(dates):
#     try:
#         rankings = pd.concat([rankings, daily_rankings(date)])
#     except (ValueError, IndexError):
#         continue
#
# rankings.to_csv('daily_rankings_raw.csv', index=False)
# # rankings = pd.concat([first, second])
#
# # rankings.reset_index(drop=True, inplace=True)
# # rankings = rankings[~rankings.duplicated()]
# #
#
# rankings = pd.read_csv('daily_rankings_raw.csv')
# indices = [i for i, row in tqdm(rankings.iterrows(), total=rankings.shape[0]) if 'Division' in row['TEAM']]
# rankings.drop(indices, axis=0, inplace=True)
# rankings.RANK = rankings.RANK.astype('float64')
# rankings.RANK = rankings.groupby('DATE').RANK.rank(method='min',ascending=False)
# rankings = rankings.sort_values(by=['DATE','RANK'], ascending=[True, False])
# rankings.TEAM = rankings.TEAM.str.extract('([A-Za-z\s\d]+)')
#
# from data.scripts.helpers import *
#
# rankings.TEAM = rankings.TEAM.apply(fix_team_names)
# rankings.reset_index(drop=True, inplace=True)
# rankings.to_csv('daily_rankings_cleaned.csv', index=False)


final.GAME_DATE1 = final.GAME_DATE1.dt.date.apply(lambda x: x.strftime("%Y-%m-%d"))
rankings = pd.read_csv("daily_rankings_cleaned.csv")
final = final.merge(
    rankings, left_on=["GAME_DATE1", "TEAM11"], right_on=["DATE", "TEAM"]
).merge(rankings, left_on=["GAME_DATE1", "TEAM22"], right_on=["DATE", "TEAM"])
final.drop(
    ["TEAM_x", "DATE_x", "TEAM_y", "DATE_y", "W_x", "L_x", "W_y", "L_y"],
    axis=1,
    inplace=True,
)
final = final.rename(
    columns={
        "RANK_x": "RANK1",
        "RANK_y": "RANK2",
        "PW_x": "PW1",
        "PL_x": "PL1",
        "PS/G_x": "PS_G1",
        "PA/G_x": "PA/G1",
        "PW_y": "PW2",
        "PL_y": "PL2",
        "PS/G_y": "PS_G2",
        "PA/G_y": "PA/G2",
    }
)

final.head()


cols = (
    ["SEASON", "GAME_DATE", "HOME_TEAM", "HOME_WL"]
    + [
        "HOME_" + re.findall("[a-zA-Z]+", col)[0]
        for col in final.columns
        if col[-1] == "1"
        and col
        not in [
            "SEASON1",
            "RANK1",
            "TEAM11",
            "GAME_DATE1",
            "WL11",
            "PW1",
            "PL1",
            "PS_G1",
            "PA/G1",
        ]
    ]
    + ["AWAY_TEAM", "AWAY_WL"]
    + [
        "AWAY_" + re.findall("[a-zA-Z]+", col)[0]
        for col in final.columns
        if col[-1] == "2"
        and col
        not in ["SEASON1", "RANK2", "TEAM22", "WL22", "PW2", "PL2", "PS_G2", "PA/G2"]
    ]
    + [
        "HOME_RANK",
        "HOME_PW",
        "HOME_PL",
        "HOME_PS_G",
        "HOME_PA/G",
        "AWAY_RANK",
        "AWAY_PW",
        "AWAY_PL",
        "AWAY_PS_G",
        "AWAY_PA/G",
    ]
)
final.columns = cols
final = final.rename(
    columns={
        "HOME_PM": "HOME_3PM",
        "HOME_PA": "HOME_3PA",
        "AWAY_PM": "AWAY_3PM",
        "AWAY_PA": "AWAY_3PA",
        "HOME_RANK": "HOME_RANK",
        "AWAY_RANK": "AWAY_RANK",
    }
)


# team offensive/defensive ratings
# 100*((Points)/(POSS) OFFENSIVE
# 100*((Opp Points)/(Opp POSS)) DEFENSIVE
# OFFRTG - DEFRTG NET
# POSS = (FGA – OREB) + TOV + (.44 * FTA)
final['HOME_POSS'] = (final['HOME_FGA'] - final['HOME_OREB']) + final['HOME_TOV'] + (0.44 * final['HOME_FTA'])
final['AWAY_POSS'] = (final['AWAY_FGA'] - final['AWAY_OREB']) + final['AWAY_TOV'] + (0.44 * final['AWAY_FTA'])

final['HOME_OFF_RATING'] = 100*(final['HOME_PTS'] / final['HOME_POSS'])
final['AWAY_OFF_RATING'] = 100*(final['AWAY_PTS'] / final['AWAY_POSS'])

final['HOME_DEF_RATING'] = 100*(final['AWAY_PTS']/final['AWAY_POSS'])
final['AWAY_DEF_RATING'] = 100*(final['HOME_PTS']/final['HOME_POSS'])


home_cols = [
    col
    for col in final.columns
    if "HOME" in col
    and col
    not in [
        "HOME_GP",
        "HOME_GAME",
        "HOME_ELO",
        "HOME_W",
        "HOME_L",
        "HOME_WL",
        "HOME_TEAM",
        "HOME_MIN",
        "HOME_RANK",
        "HOME_PW",
        "HOME_PL",
        "HOME_PS_G",
        "HOME_PA/G",
       "HOME_POSS",
        "HOME_OFF_RATING",
       "HOME_DEF_RATING",
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
        "AWAY_RANK",
        "AWAY_PW",
        "AWAY_PL",
        "AWAY_PS_G",
        "AWAY_PA/G",
        "AWAY_POSS",
         "AWAY_OFF_RATING",
         "AWAY_DEF_RATING",
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
final.drop(["HOME_GP", "AWAY_GP", "GAME_DATE"], axis=1, inplace=True)
final = pd.get_dummies(final, drop_first=True)
final = final.rename(columns={"HOME_WL_W": "OUTCOME"})  # 1 if home team wins
final.reset_index(drop=True, inplace=True)


outcome = final["OUTCOME"]
first = final[[col for col in final.columns if "HOME" in col]]
second = final[[col for col in final.columns if "AWAY" in col]]

final = first.div(second.values)
final.columns = [col.split("_")[-1] + "_RATIO" for col in final.columns]
final["OUTCOME"] = outcome
final.replace([np.inf, -np.inf], np.nan, inplace=True)

final.isnull().sum()
final = final.dropna()
final.reset_index(drop=True, inplace=True)
final.columns = ['ELO_RATIO', 'W_RATIO', 'L_RATIO', 'FGM_RATIO', 'FGA_RATIO',
                 '3PM_RATIO', '3PA_RATIO', 'FTM_RATIO', 'FTA_RATIO', 'OREB_RATIO',
                 'DREB_RATIO', 'REB_RATIO', 'AST_RATIO', 'TOV_RATIO', 'STL_RATIO',
                 'BLK_RATIO', 'PF_RATIO', 'RANK_RATIO', 'PW_RATIO', 'PL_RATIO',
                 'PS/G_RATIO', 'PA/G_RATIO', 'POSS_RATIO','OFF_RATING_RATIO',
                 'DEF_RATING_RATIO', 'OUTCOME']
final.head()
final.shape

# Machine Learning
# from data.scripts.helpers import FastML
#
# ml = FastML("classification", final, "OUTCOME")
# ml.results()

#                     Log-Loss  F1-Score       ROC
# Catboost            0.629497  0.731291  0.673738
# LightGBM            0.630528  0.729459  0.671961
# RandomForests       0.634822  0.723299  0.665381
# LogisticRegression  0.641086  0.730727  0.651631
# XGBoost             0.672537  0.706751  0.644954
# KNN                 2.059518  0.669922  0.592626

# Hyperparameter
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from lightgbm import early_stopping
from verstack import LGBMTuner

X = final.drop("OUTCOME", axis=1)
y = final["OUTCOME"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42)

# Model ve Hiperparametre
tuner = LGBMTuner(
    metric="f1_weighted", trials=1000, refit=True, verbosity=5, visualization=True, seed=42
)
tuner.fit(X_train, y_train)


# Final Model
model_tuned = lgbm.LGBMClassifier(**tuner.best_params).fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="f1_weighted",
    verbose=1,
    callbacks=[early_stopping(100)],
)

y_pred = model_tuned.predict(X_test)
f1_score(y_test, y_pred)  # 0.754
confusion_matrix(y_test, y_pred)
# [ 470, 1532]
# [ 239, 2715]
print(classification_report(y_test, y_pred))
#               precision    recall  f1-score   support
#          0.0       0.61      0.35      0.45      2002
#          1.0       0.66      0.85      0.74      2954
#     accuracy                           0.65      4956
#    macro avg       0.64      0.60      0.59      4956
# weighted avg       0.64      0.65      0.62      4956
proba_pred = model_tuned.predict_proba(X_test)
proba_df = pd.DataFrame(proba_pred)
X_test.reset_index(drop=True, inplace=True)
X_test["0"] = proba_df[0]
X_test["1"] = proba_df[1]
X_test["Real"] = y
X_test["Pred"] = y_pred
X_test[["0", "1", "Real", "Pred"]].iloc[50:100]

import matplotlib.pyplot as plt
import seaborn as sns

final.OUTCOME.value_counts()

sns.histplot(X_test["1"], kde=True)
plt.show()

