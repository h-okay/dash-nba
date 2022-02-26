import pandas as pd
import numpy as np
from datetime import date
from tqdm import tqdm
from data.scripts.helpers import fix_team_names

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)

teams = pd.read_csv("data/base/all_teams.csv")

################# MVP CANDIDATES TABLE ##############
year = date.today().year
# Getting ALL MVP candidates year by year from 2004
mvp_cand = pd.DataFrame()
advanced = pd.DataFrame()
team_standings = pd.DataFrame()
for i in tqdm(range(1980, year)):
    temp1 = pd.read_html(f"https://www.basketball-reference.com/awards/awards_{i}.html", header=1, match="Most Valuable Player")[0]
    temp1["Year"] = str(i)
    mvp_cand = mvp_cand.append(temp1, ignore_index=True)

    temp2 = pd.read_html(f"https://www.basketball-reference.com/leagues/NBA_{i}_advanced.html", header=0, match="Advanced")[0]
    temp2.drop(["Unnamed: 19", "Unnamed: 24"], axis=1, inplace=True)
    temp2["Player"] = temp2["Player"].str.replace("*", "")
    temp2["Year"] = str(i)
    advanced = advanced.append(temp2, ignore_index=True)

    temp3_e = pd.read_html(f"https://www.basketball-reference.com/leagues/NBA_{i}_standings.html", header = 0)[0]
    temp3_e["Year"] = str(i)
    temp3_e = temp3_e.rename({"Eastern Conference" : "Team"}, axis=1)
    temp3_e = temp3_e[temp3_e['Team'].str.contains('Division') == False]

    temp3_w = pd.read_html(f"https://www.basketball-reference.com/leagues/NBA_{i}_standings.html", header = 0)[1]
    temp3_w["Year"] = str(i)
    temp3_w = temp3_w.rename({"Western Conference": "Team"}, axis=1)
    temp3_w = temp3_w[temp3_w['Team'].str.contains('Division') == False]

    temp3 = pd.concat([temp3_e, temp3_w], ignore_index=True)
    temp3["Team"] = temp3["Team"].str.replace("*", "")
    team_standings = team_standings.append(temp3)

team_standings["Team"] = team_standings.Team.apply(fix_team_names)
team_standings.tail(20)
teams.tail(20)
teams["full_name"] = teams.full_name.apply(fix_team_names)

team_standings = pd.merge(team_standings, teams, how="left", left_on="Team", right_on="full_name")
team_standings.tail(20)
team_standings.rename(columns={"abbreviation" : "Tm"}, inplace=True)

all_mvp_cand = pd.merge(mvp_cand, advanced, how="left", on=["Player", "Tm", "Year"])

team_standings[(team_standings["Tm"] == "MIL")]
all_mvp_cand[(all_mvp_cand["Tm"] == "MIL") & (all_mvp_cand["Year"] == "2021")]

all_mvp_cand = pd.merge(all_mvp_cand, team_standings, how="left", on=["Tm", "Year"])
all_mvp_cand["MVP"] = np.where((all_mvp_cand["Rank"] == "1"), 1, 0)

all_mvp_cand = all_mvp_cand.drop(["Rk", "Age_y", "G_y", "MP_y", "WS_y", "WS/48_y", "W", "L", "GB", "PS/G", "PA/G", "SRS",
                                  "id", "full_name", "nickname", "city", "state", "year_founded"], axis=1)


all_mvp_cand.rename(columns={ "Age_x" : "Age",
                           "WS_x" : "WS",
                          "WS/48_x" : "WS/48",
                          "G_x" : "G",
                          "MP_x" : "MP"}, inplace=True)



all_mvp_cand.to_csv("data/base/mvp_candidates.csv", index=False)






