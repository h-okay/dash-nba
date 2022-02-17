import pandas as pd
from data.scripts.helpers import *
import pickle as pkl
import numpy as np
from time import sleep

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service

from data.scripts.helpers import *

pd.set_option('display.max_columns', None)

mlready = pd.read_csv('data/est/mlready.csv')

filter = mlready[mlready.SEASON == '2021-22']
filter.columns

filter['FG%'] = filter['FGM'] / filter['FGA']
filter['FG3%'] = filter['FG3M'] / filter['FG3A']
filter['FT%'] = filter['FTM'] / filter['FTA']

filter.drop(['FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB',
             'Playoff','Quarter','Semi','Final','Champion','GB','CONF',
             'DIV','OT','TEAM_ID', 'SCORE', 'SEASON', 'LAST 10',
             'STREAK'], axis=1, inplace=True)


filter.head()

filter['HOME_WIN'] = filter.HOME.apply(lambda x: int(x.split("-")[0]))
filter['HOME_LOSE'] = filter.HOME.apply(lambda x: int(x.split("-")[1]))
filter['AWAY_WIN'] = filter.ROAD.apply(lambda x: int(x.split("-")[0]))
filter['AWAY_LOSE'] = filter.ROAD.apply(lambda x: int(x.split("-")[0]))
################################
with open("data/base/matches.pkl", "rb") as file:
    matches = pkl.load(file)

for i in matches.values():
    cols = i.columns
    break

starter = pd.DataFrame(columns=cols)
for value in matches.values():
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

###########

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
elo_ratings = np.full(len(team_ids), 1400)  # Takımların başlangıç elolarını belirt.
a = concat_matches[["TEAM_ID_x", "GAME_DATE_x"]]
checkpoint = concat_matches[~a.duplicated()]  # 6 maç drop oldu.
checkpoint = (
    checkpoint.sort_values("GAME_DATE_x").reset_index().drop("index", axis=1)
)

elo_dict = dict(zip(team_ids, elo_ratings))
elo_date_team = pd.DataFrame(columns=["DATE", "SEASON", "TEAM_ID", "ELO"])


def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

# Maçın sonucuna göre takımın yeni ELOsunu bul.
def new_rating(elo_a, score_a, expected_score_a):
    return elo_a + 32 * (score_a - expected_score_a)

# Elo hesaplama döngüsü
for index, row in tqdm(checkpoint.iterrows(), total=checkpoint.shape[0]):
    # print(f"{row['TEAM_NAME_x']} vs. {row['TEAM_NAME_y']} @ {row['GAME_DATE_x']}")
    season_id = row["SEASON_ID_x"]
    game_date = row["GAME_DATE_x"]
    home_id = row["TEAM_ID_x"]
    home_elo = elo_dict[home_id]
    away_id = row["TEAM_ID_y"]
    away_elo = elo_dict[away_id]
    exp_scr = expected_score(home_elo, away_elo)
    result = row["WL_x"]

    if result == "W":
        elo_dict[home_id] = new_rating(home_elo, 1, exp_scr)
        elo_dict[away_id] = new_rating(away_elo, 0, 1 - exp_scr)
    elif result == "L":
        elo_dict[home_id] = new_rating(home_elo, 0, exp_scr)
        elo_dict[away_id] = new_rating(away_elo, 1, 1 - exp_scr)
    else:
        print(f"{season_id} season, {game_date} dated match is defected.")
        continue

    elo_date_team = elo_date_team.append(
        {
            "DATE": game_date,
            "SEASON": season_id,
            "TEAM_ID": home_id,
            "ELO": elo_dict[home_id],
        },
        ignore_index=True,
    )
    elo_date_team = elo_date_team.append(
        {
            "DATE": game_date,
            "SEASON": season_id,
            "TEAM_ID": away_id,
            "ELO": elo_dict[away_id],
        },
        ignore_index=True,
    )

edt = elo_date_team.copy()
edt = edt.sort_values(by="DATE")
edt.to_csv('save_elo_ts.csv', index=False)

all_teams = pd.read_csv('data/base/all_teams.csv')
all_teams = all_teams[['id', 'full_name']]

all_data = pd.DataFrame()
timeout = 5
ser = Service("C:/Program Files/chromedriver.exe")
s = webdriver.Chrome(service=ser)
for id in all_teams.id.unique():
    for season in edt.SEASON.unique():
        random = np.random.random() * 3
        url = (
            f"https://www.nba.com/stats/team/{id}/traditional/?Season={season}&SeasonType=Regular%20Season&PerMode=Totals"
        )
        s.get(url)
        element_present = EC.presence_of_element_located((By.CLASS_NAME, "nba-stat-table__overflow"))
        WebDriverWait(s, timeout).until(element_present)
        html = s.page_source
        tables = pd.read_html(html)
        data = tables[6]
        data['TEAM'] = all_teams[all_teams.id == id].full_name.values[0]
        data["SEASON"] = season
        all_data = pd.concat([all_data, data], axis=0, ignore_index=True)
        sleep(.6 + random)

s.quit()


all_data = pd.concat([first, second, third, fourth, fifth])
all_data.to_csv('monthly_team_perf.csv', index=False)


