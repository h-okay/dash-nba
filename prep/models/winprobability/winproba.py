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

# from utils.helpers import FastML

#Get Monthly Performances / Selenium
# Ay ay takımların o sezonki performanslarının alınması. Sadece o ay içinde oynanan
# maçlardaki performansları. Sezon ilerledikçe cumulative olarak artan istatistikler değil.
# Örn. Ocak ayı 5 maç oynanmışken şubat 3 maç oynandı gibi(cumulativ olsaydı şubat 8 olmalıydı.)
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

# Matches pickle dosyasının zaman serisine çevrilmesi. Elo hesabunda da aynı format kullanılmıştı
def get_matches():
    with open("prep/data/matches.pkl", "rb") as file:
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
all_teams = pd.read_csv("prep/data/all_teams.csv")
all_teams = all_teams[["id", "full_name"]]
elo_ts = pd.read_csv("prep/data/save_elo_ts.csv")
df = (
    elo_ts.merge(all_teams, left_on="TEAM_ID", right_on="id")
    .drop("id", axis=1)
    .rename(columns={"full_name": "TEAM"})
)
df = df.sort_values(by="DATE").reset_index(drop=True)
df.DATE = pd.to_datetime(df.DATE)
df["MONTH"] = df.DATE.dt.month_name()
df.head() # Takımların o ayki elolarını aylık performans tablasuna merge etme.
# # Buradaki problem bir takımın o ay içinde oynadığı kadar elo değeri geliyor.
# Bunun önüne geçmek için aylık sonraki aşamada ortalama alıyoruz.
#         DATE   SEASON     TEAM_ID     ELO                  TEAM    MONTH
# 0 2003-10-05  2003-04  1610612762  1416.0             Utah Jazz  October
# 1 2003-10-05  2003-04  1610612742  1384.0      Dallas Mavericks  October
# 2 2003-10-06  2003-04  1610612763  1416.0     Memphis Grizzlies  October
# 3 2003-10-06  2003-04  1610612749  1384.0       Milwaukee Bucks  October
# 4 2003-10-07  2003-04  1610612746  1384.0  Los Angeles Clippers  October
df.tail()

# Get monthly averages / Aylık ortalamaların alınması
agg_df = df.groupby(["SEASON", "TEAM", "MONTH"]).ELO.mean().reset_index()
agg_df[
    (agg_df.SEASON == "2021-22") & (agg_df.TEAM == "Toronto Raptors")
].MONTH.value_counts()  # 1 for each month as expected

# Data leakage engellemek için aylık değerleri 1 birim kaydırmamız gerek.
# Yani ocak ayı satırının yanında aralık ayı eloları yazmalı.
monthly_elo = pd.DataFrame()  # 1 month shifted for ML purposes / prevent data leakeage
for team in agg_df.TEAM.unique():
    temp = agg_df[(agg_df.TEAM == team)]
    temp.MONTH = temp.MONTH.shift(1)
    monthly_elo = pd.concat([monthly_elo, temp])

monthly_elo.isnull().sum()  # 30 as expected
monthly_elo.dropna(inplace=True)


monthly = pd.read_csv("prep/data/monthly_team_perf.csv")
monthly = monthly.rename(columns={"Month": "MONTH"})
monthly.drop(["WIN%", "FG%", "3P%", "FT%", "+/-"], axis=1, inplace=True)
monthly.head()
#       MONTH  GP  MIN   PTS  W   L  FGM   FGA  3PM  3PA  FTM  FTA  OREB  DREB  \
# 0   October   2  101   177  0   2   69   173   10   26   29   40    23    83
# 1  November  16  778  1454  6  10  544  1251   71  231  295  370   185   495
# 2  December  15  720  1343  3  12  530  1195   49  162  234  318   181   476
# 3   January  15  720  1267  6   9  484  1111   51  156  248  319   160   442
# 4  February  11  543  1044  4   7  392   920   71  204  189  265   145   344
#    REB  AST  TOV  STL  BLK   PF           TEAM   SEASON
# 0  106   40   42   13   18   49  Atlanta Hawks  2003-04
# 1  680  316  277  118   97  339  Atlanta Hawks  2003-04
# 2  657  301  241   96   74  336  Atlanta Hawks  2003-04
# 3  602  291  262  113   73  348  Atlanta Hawks  2003-04
# 4  489  219  185  100   70  239  Atlanta Hawks  2003-04
monthly.tail()

# Yukarıda bahsedildiği gibi aylık istatistikler sadece o ay için geçerli. Bu sebeple rolling sum almalıyız.
# Yani o aya kadar ortaya çıkan tüm istatistikler o satırda bulunmalı.
# rolling sum nasıl çalışır?
# 1-3-4-8-16 --> hep kendinden öncekilerin tümünü toplayarak devam eder.
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

# Yine data leakage engellemek için bu istatistikleride 1 ay kaydırmalıyız.
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

# matches time series ile hazırlanın verinin birleştirilmesi (maç skorlarını ve sonuçlarını almak adına)
melo = monthly_elo.merge(lagged_stats, on=["SEASON", "MONTH", "TEAM"])
melo.to_csv("prep/models/winprobability/data/melo.csv", index=False)
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
#    SEASON1 GAME_DATE1                  TEAM11 WL11         ELO1   GP1    MIN1  \
# 0  2003-04 2003-11-01           Orlando Magic    L  1376.724564  32.0  1546.0
# 1  2003-04 2003-11-01   Golden State Warriors    W  1383.295974  30.0  1450.0
# 2  2003-04 2003-11-01  Minnesota Timberwolves    W  1406.913239  30.0  1455.0
# 3  2003-04 2003-11-01        Dallas Mavericks    W  1392.711282  30.0  1445.0
# 4  2003-04 2003-11-01         Milwaukee Bucks    W  1380.107642  31.0  1498.0
#      PTS1    FGM1    FGA1   3PM1   3PA1   FTM1   FTA1  OREB1  DREB1    REB1  \
# 0  2941.0  1108.0  2659.0  155.0  461.0  570.0  760.0  419.0  930.0  1349.0
# 1  2777.0  1067.0  2392.0  143.0  466.0  500.0  708.0  342.0  955.0  1297.0
# 2  2849.0  1146.0  2410.0  103.0  298.0  454.0  590.0  284.0  945.0  1229.0
# 3  3058.0  1158.0  2645.0  194.0  581.0  548.0  705.0  464.0  941.0  1405.0
# 4  2932.0  1105.0  2528.0  153.0  444.0  569.0  755.0  364.0  963.0  1327.0
#     AST1   TOV1   STL1   BLK1    PF1              TEAM22 WL22         ELO2  \
# 0  603.0  452.0  215.0  148.0  677.0     Detroit Pistons    W  1392.861893
# 1  633.0  448.0  203.0  113.0  621.0  Philadelphia 76ers    L  1390.791157
# 2  717.0  387.0  213.0  177.0  648.0     Toronto Raptors    L  1432.435477
# 3  671.0  383.0  224.0  138.0  612.0           Utah Jazz    L  1424.000000
# 4  713.0  439.0  211.0  174.0  637.0       Chicago Bulls    L  1379.156999
#     GP2    MIN2    PTS2    FGM2    FGA2   3PM2   3PA2   FTM2   FTA2  OREB2  \
# 0  32.0  1546.0  2829.0  1045.0  2462.0  111.0  331.0  628.0  821.0  417.0
# 1  32.0  1546.0  2830.0  1045.0  2439.0  145.0  388.0  595.0  791.0  371.0
# 2  30.0  1465.0  2579.0   971.0  2285.0  160.0  434.0  477.0  632.0  297.0
# 3  31.0  1498.0  2848.0  1067.0  2342.0   92.0  275.0  622.0  842.0  417.0
# 4  31.0  1493.0  2760.0  1050.0  2529.0  153.0  446.0  507.0  686.0  396.0
#    DREB2    REB2   AST2   TOV2   STL2   BLK2    PF2
# 0  939.0  1356.0  599.0  499.0  241.0  214.0  676.0
# 1  880.0  1251.0  655.0  493.0  294.0  131.0  658.0
# 2  895.0  1192.0  614.0  454.0  199.0  146.0  617.0
# 3  866.0  1283.0  658.0  532.0  214.0  194.0  783.0
# 4  944.0  1340.0  681.0  507.0  260.0  151.0  694.0

# Adding rankings before match as features
import re

# feature engineering için maçın oynandığı günden bir önceki gündeki genel sıralamarını
# ekleyelim.
def daily_rankings(date):
    year = date.split("-")[0]
    month = date.split("-")[1]
    day = date.split("-")[2]
    url = f"https://www.basketball-reference.com/friv/standings.fcgi?month={month}&day={day}&year={year}&lg_id=NBA"
    # Atlantic-Central
    ac = pd.read_html(url)[0]
    # ac.drop([0, 8], axis=0, inplace=True)
    ac.drop("GB", axis=1, inplace=True)
    ac.columns = ["TEAM", "W", "L", "RANK", "PW", "PL", "PS/G", "PA/G"]
    # Midwest-Pacific
    mp = pd.read_html(url)[1]
    # mp.drop([0, 8], axis=0, inplace=True)
    mp.drop("GB", axis=1, inplace=True)
    mp.columns = ["TEAM", "W", "L", "RANK", "PW", "PL", "PS/G", "PA/G"]
    acmp = pd.concat([ac, mp], axis=0)
    if len(str(day)) == 1:
        acmp["DATE"] = str(year) + "-" + str(month) + "-0" + str(day)
    else:
        acmp["DATE"] = str(year) + "-" + str(month) + "-" + str(day)
    return acmp


dates = final.GAME_DATE1.dt.date.apply(lambda x: x.strftime("%Y-%m-%d")).unique()

rankings = pd.DataFrame()
for date in tqdm(dates):
    try:
        rankings = pd.concat([rankings, daily_rankings(date)])
    except (ValueError, IndexError):
        continue

rankings.to_csv("daily_rankings_raw.csv", index=False)

rankings = pd.read_csv("data/prep/daily_rankings_raw.csv")
indices = [
    i
    for i, row in tqdm(rankings.iterrows(), total=rankings.shape[0])
    if "Division" in row["TEAM"]
]
rankings.drop(indices, axis=0, inplace=True)
rankings.RANK = rankings.RANK.astype("float64")
rankings.RANK = rankings.groupby("DATE").RANK.rank(method="min", ascending=False)
rankings = rankings.sort_values(by=["DATE", "RANK"], ascending=[True, False])
rankings.TEAM = rankings.TEAM.str.extract("([A-Za-z\s\d]+)")

from data.scripts.helpers import *

rankings.TEAM = rankings.TEAM.apply(fix_team_names)
rankings.reset_index(drop=True, inplace=True)
rankings.to_csv("daily_rankings_cleaned.csv", index=False)

# sıralamalar ile final df 'in birleştirilmesi
final.GAME_DATE1 = final.GAME_DATE1.dt.date.apply(lambda x: x.strftime("%Y-%m-%d"))
rankings = pd.read_csv("prep/data/daily_rankings_cleaned.csv")
final = final.merge(
    rankings, left_on=["GAME_DATE1", "TEAM11"], right_on=["DATE", "TEAM"]
).merge(rankings, left_on=["GAME_DATE1", "TEAM22"], right_on=["DATE", "TEAM"])
final.drop(
    [
        "TEAM_x",
        "DATE_x",
        "TEAM_y",
        "DATE_y",
        "PW_x",
        "PW_y",
        "PL_x",
        "PL_y",
        "PS/G_x",
        "PA/G_x",
        "PS/G_y",
        "PA/G_y",
    ],
    axis=1,
    inplace=True,
)
final = final.rename(
    columns={
        "RANK_x": "RANK1",
        "RANK_y": "RANK2",
        "W_x": "WINS1",
        "W_y": "WINS2",
        "L_x": "LOSS1",
        "L_y": "LOSS2",
    }
)

cols = (
    ["SEASON", "GAME_DATE", "HOME_TEAM", "HOME_WL"]
    + [
        "HOME_" + re.findall("[a-zA-Z]+", col)[0]
        for col in final.columns
        if col[-1] == "1"
        and col
        not in ["SEASON1", "RANK1", "TEAM11", "GAME_DATE1", "WL11", "WINS1", "LOSS1"]
    ]
    + ["AWAY_TEAM", "AWAY_WL"]
    + [
        "AWAY_" + re.findall("[a-zA-Z]+", col)[0]
        for col in final.columns
        if col[-1] == "2"
        and col not in ["SEASON1", "RANK2", "TEAM22", "WL22", "WINS2", "LOSS2"]
    ]
    + [
        "HOME_WINS",
        "HOME_LOSS",
        "HOME_RANK",
        "AWAY_WINS",
        "AWAY_LOSS",
        "AWAY_RANK",
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

# team offensive/defensive ratings - feature engineering- takımın o maç öncesi performansına göre
# possession(maç başına kullanılan ortalama top)
# offensive ve defensive ratingler
# kazanma yüzdesi ekleyelim.

# 100*((Points)/(POSS) OFFENSIVE
# 100*((Opp Points)/(Opp POSS)) DEFENSIVE
# OFFRTG - DEFRTG NET
# POSS = (FGA – OREB) + TOV + (.44 * FTA)
final["HOME_POSS"] = (
    (final["HOME_FGA"] - final["HOME_OREB"])
    + final["HOME_TOV"]
    + (0.44 * final["HOME_FTA"])
)
final["AWAY_POSS"] = (
    (final["AWAY_FGA"] - final["AWAY_OREB"])
    + final["AWAY_TOV"]
    + (0.44 * final["AWAY_FTA"])
)

final["HOME_OFF_RATING"] = 100 * (final["HOME_PTS"] / final["HOME_POSS"])
final["AWAY_OFF_RATING"] = 100 * (final["AWAY_PTS"] / final["AWAY_POSS"])

final["HOME_DEF_RATING"] = 100 * (final["AWAY_PTS"] / final["AWAY_POSS"])
final["AWAY_DEF_RATING"] = 100 * (final["HOME_PTS"] / final["HOME_POSS"])

final["HOME_GP"] = final["HOME_WINS"] + final["HOME_LOSS"]
final["AWAY_GP"] = final["AWAY_WINS"] + final["AWAY_LOSS"]

final["HOME_WIN_PERC"] = (final["HOME_WINS"] / final["HOME_GP"] + 0.1) * 100
final["AWAY_WIN_PERC"] = (final["AWAY_WINS"] / final["AWAY_GP"] + 0.1) * 100

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
        "HOME_POSS",
        "HOME_OFF_RATING",
        "HOME_DEF_RATING",
        "HOME_WINS",
        "HOME_LOSS",
        "HOME_NET_RATING",
        "HOME_PACE",
        "HOME_WIN_PERC",
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
        "AWAY_POSS",
        "AWAY_OFF_RATING",
        "AWAY_DEF_RATING",
        "AWAY_WINS",
        "AWAY_LOSS",
        "AWAY_NET_RATING",
        "AWAY_PACE",
        "AWAY_WIN_PERC",
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
        "HOME_GP",
        "AWAY_GP",
        "GAME_DATE",
    ],
    axis=1,
    inplace=True,
)
final = pd.get_dummies(final, drop_first=True)
final = final.rename(columns={"HOME_WL_W": "OUTCOME"})  # 1 if home team wins
# ev sahibinin kazanması durumuna 1 aksi duruma 0 diyerek binary classification problemi olarak yaklaşalım
final.reset_index(drop=True, inplace=True)

outcome = final["OUTCOME"]
first = final[[col for col in final.columns if "HOME" in col]]
second = final[[col for col in final.columns if "AWAY" in col]]


# elde edilen istatistikleri birbirlerine oranlayarak o maçı oynayan takımlardan ratio'lar elde edelim.
final = first.div(second.values)
final.columns = [col.split("_")[-1] + "_RATIO" for col in final.columns]
final["OUTCOME"] = outcome
final.replace([np.inf, -np.inf], np.nan, inplace=True)
final.dropna(inplace=True)
final.reset_index(drop=True, inplace=True)
final.columns = [
    "ELO_RATIO",
    "PTS_RATIO",
    "FGM_RATIO",
    "FGA_RATIO",
    "3PM_RATIO",
    "3PA_RATIO",
    "FTM_RATIO",
    "FTA_RATIO",
    "OREB_RATIO",
    "DREB_RATIO",
    "REB_RATIO",
    "AST_RATIO",
    "TOV_RATIO",
    "STL_RATIO",
    "BLK_RATIO",
    "PF_RATIO",
    "WINS_RATIO",
    "LOSS_RATIO",
    "RANK_RATIO",
    "POSS_RATIO",
    "OFF_RATING_RATIO",
    "DEF_RATING_RATIO",
    "WIN_PERC_RATIO",
    "OUTCOME",
]

# FINAL DATA
#        ELO_RATIO  PTS_RATIO  FGM_RATIO  FGA_RATIO  3PM_RATIO  3PA_RATIO  \
# 0       0.988414   1.039590   1.060287   1.080016   1.396396   1.392749
# 1       0.994611   1.471908   1.531579   1.471095   1.479310   1.801546
# 2       0.982183   1.104692   1.180227   1.054705   0.643750   0.686636
# 3       0.978028   0.715824   0.723524   0.752918   1.405797   1.408485
# 4       1.000689   1.062319   1.052381   0.999605   1.000000   0.995516
#           ...        ...        ...        ...        ...        ...
# 19267   0.862969   0.921073   0.912805   1.033055   0.908123   1.027856
# 19268   0.973168   0.981473   0.970640   1.031117   0.993548   1.057571
# 19269   0.969316   0.994842   1.004108   1.002063   1.011461   1.042490
# 19270   1.144776   0.993015   0.993342   0.960578   0.998390   0.919516
# 19271   0.809896   0.959558   0.906043   0.923506   1.203243   1.195146
#        FTM_RATIO  FTA_RATIO  OREB_RATIO  DREB_RATIO  REB_RATIO  AST_RATIO  \
# 0       0.907643   0.925700    1.004796    0.990415   0.994838   1.006678
# 1       1.260504   1.342604    1.382749    1.627841   1.555156   1.449618
# 2       0.951782   0.933544    0.956229    1.055866   1.031040   1.167752
# 3       0.587353   0.558195    0.741807    0.724403   0.730060   0.679838
# 4       1.122288   1.100583    0.919192    1.020127   0.990299   1.046990
#           ...        ...         ...         ...        ...        ...
# 19267   0.977117   1.012647    1.151970    1.022321   1.049431   0.790712
# 19268   1.022541   1.109440    1.084936    1.039206   1.050232   0.852921
# 19269   0.939571   0.944817    1.091803    0.997514   1.019458   1.051429
# 19270   0.988338   0.968291    0.738998    0.986147   0.923696   0.916318
# 19271   1.071522   1.098434    0.709218    0.940933   0.875259   0.910763
#        TOV_RATIO  STL_RATIO  BLK_RATIO  PF_RATIO  WINS_RATIO  LOSS_RATIO  \
# 0       0.905812   0.892116   0.691589  1.001479    0.500000    2.000000
# 1       1.363083   1.035714   1.293893  1.415653    1.000000    0.500000
# 2       0.852423   1.070352   1.212329  1.050243    1.000000    1.000000
# 3       0.479950   0.697819   0.474227  0.521073    2.000000    1.000000
# 4       0.865878   0.811538   1.152318  0.917867    2.000000    0.500000
#           ...        ...        ...       ...         ...         ...
# 19267   0.938776   1.048309   1.267606  0.944395    0.535714    1.619048
# 19268   0.926999   0.935927   1.602151  0.984087    0.750000    1.421053
# 19269   1.011765   0.960739   1.192029  1.067132    1.055556    0.970588
# 19270   0.849462   1.004598   1.293173  0.947781    1.722222    0.593750
# 19271   1.279950   0.714234   0.744789  1.048202    0.400000    2.000000
#        RANK_RATIO  POSS_RATIO  OFF_RATING_RATIO  DEF_RATING_RATIO  \
# 0        5.000000    1.041704          0.997971          1.002033
# 1        0.750000    0.965789          1.016031          0.984222
# 2        1.000000    1.019308          1.083766          0.922708
# 3        0.266667    1.016524          1.056282          0.946717
# 4        0.200000    0.997743          1.064722          0.939212
#            ...         ...               ...               ...
# 19267    2.454545    1.006913          0.914749          1.093196
# 19268    3.800000    1.016736          0.965317          1.035929
# 19269    0.923077    0.989036          1.005870          0.994164
# 19270    0.240000    0.970484          1.023216          0.977311
# 19271    9.333333    0.958109          0.944822          1.058400
#        WIN_PERC_RATIO  OUTCOME
# 0            0.565217        0
# 1            1.384615        1
# 2            1.000000        1
# 3            1.277778        1
# 4            1.769231        1
#                ...      ...
# 19267        0.604863        1
# 19268        0.784367        1
# 19269        1.043103        1
# 19270        1.565217        1
# 19271        0.499752        0


final.OUTCOME.value_counts()  # Imbalance
# 1    11359
# 0     7783

X = final.drop("OUTCOME", axis=1)
y = final["OUTCOME"]


# Classification classları arasında 12000'e 8000 gibi bir dengesizlik var.
# Oversampling ile 2 grubu da eşitleyelim.
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(sampling_strategy=1)
X, y = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42
)

df = pd.concat([X_train, y_train], axis=1)
ml = FastML("classification", df, "OUTCOME", refit=False)
# Hızlıca sonuçlar hakkında  fikir almak için bir kaç model deneyelim.
best_model = ml.results()
#                     Log-Loss  F1-Score       ROC
# RandomForests       0.560754  0.697444  0.780257
# LightGBM            0.609899  0.664783  0.726643
# Catboost            0.610150  0.661676  0.726702
# XGBoost             0.620632  0.668269  0.733019
# LogisticRegression  0.653799  0.590227  0.663690
# KNN                 2.013682  0.592006  0.640187

# Ligthgbm kullanılmaya kara verilmiştir.
# Hyperparameter
from verstack import LGBMTuner
from lightgbm import early_stopping

tuner = LGBMTuner(
    metric="auc", trials=100, refit=True, verbosity=5, visualization=True, seed=42
)
tuner.fit(X_train, y_train)

# Final Model
model_tuned = lgbm.LGBMClassifier(**tuner.best_params)
model_tuned.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="roc_auc",
    callbacks=[early_stopping(150)],
)

# Learning Curve --------------------------------------------------------------
from yellowbrick.model_selection import LearningCurve

visualizer = LearningCurve(model_tuned, cv=3, scoring="roc_auc", n_jobs=-1)
visualizer.fit(X_train, y_train)
visualizer.show()

# ROC Curve -------------------------------------------------------------------
from yellowbrick.classifier import ROCAUC

visualizer = ROCAUC(model_tuned, classes=[0, 1])
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

# Probability optimization
from sklearn.calibration import CalibratedClassifierCV

calibrated_clf = CalibratedClassifierCV(base_estimator=model_tuned, cv="prefit")
calibrated_clf.fit(X_calib, y_calib)

y_pred = calibrated_clf.predict_proba(X_test)
pd.DataFrame(y_pred)

# Metrics ---------------------------------------------------------------------
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    plot_roc_curve,
    confusion_matrix,
    classification_report,
)

ve, confusion_matrix, classification_report
y_pred = model_tuned.predict(X_test)
accuracy_score(y_test, y_pred)  # 0.722088
balanced_accuracy_score(y_test, y_pred)  # 0.722109
f1_score(y_test, y_pred)  # 0.719453
roc_auc_score(y_test, y_pred)  # 0.72210
confusion_matrix(y_test, y_pred)
# [1674,  610]
# [ 662, 1631]
print(classification_report(y_test, y_pred))
#               precision    recall  f1-score   support
#            0       0.72      0.73      0.72      2284
#            1       0.73      0.71      0.72      2293
#     accuracy                           0.72      4577
#    macro avg       0.72      0.72      0.72      4577
# weighted avg       0.72      0.72      0.72      4577

pd.DataFrame(
    data=model_tuned.feature_importances_,
    index=X.columns.to_list(),
    columns=["Importance"],
).sort_values(by="Importance", ascending=False)
#                   Importance
# ELO_RATIO               7685
# BLK_RATIO               7584
# RANK_RATIO              7157
# TOV_RATIO               6713
# STL_RATIO               6670
# OREB_RATIO              6660
# LOSS_RATIO              6410
# PF_RATIO                6323
# POSS_RATIO              6317
# AST_RATIO               6115
# 3PA_RATIO               6002
# FTM_RATIO               5882
# WIN_PERC_RATIO          5768
# 3PM_RATIO               5677
# WINS_RATIO              5492
# FTA_RATIO               5463
# OFF_RATING_RATIO        5402
# DREB_RATIO              5363
# FGA_RATIO               5066
# REB_RATIO               4892
# FGM_RATIO               4840
# PTS_RATIO               4512
# DEF_RATING_RATIO        4127


# EDA
proba_pred = model_tuned.predict_proba(X_test)
proba_df = pd.DataFrame(proba_pred)
X_test.reset_index(drop=True, inplace=True)
X_test["0"] = proba_df[0]
X_test["1"] = proba_df[1]
X_test["Real"] = y_test.to_list()
X_test["Pred"] = y_pred
X_test[["0", "1", "Real", "Pred"]]

for i, row in X_test.iterrows():
    if row["Real"] == row["Pred"]:
        X_test.loc[i, "True"] = "Yes"
    else:
        X_test.loc[i, "True"] = "No"

# Got right
right = X_test[X_test["True"] == "Yes"]  # 3285
right["Proba_Diff"] = right["0"] - right["1"]
right[["0", "1", "Real", "Pred"]].head(10)

# Got wrong
wrong = X_test[X_test["True"] == "No"]  # 1303
wrong["Proba_Diff"] = wrong["0"] - wrong["1"]
wrong[["0", "1", "Real", "Pred"]].head(10)

with open("winprobamodel.pkl", "wb") as file:
    pkl.dump(model_tuned, file)
