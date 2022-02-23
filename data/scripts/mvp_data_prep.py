import pandas as pd
import numpy as np
from basketball_reference_scraper import players
from datetime import date
from tqdm import tqdm
from data.scripts.helpers import fix_team_names

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)



year = date.today().year

#######
## MVP CANDIDATES TABLE
# MVP candidates year by year from 2004
mvp_cand = pd.DataFrame()
for i in tqdm(range(2004, year)):
    temp = pd.read_html(f"https://www.basketball-reference.com/awards/awards_{i}.html", header=1, match="Most Valuable Player")[0]
    temp["year"] = i
    mvp_cand = pd.concat([mvp_cand,temp], ignore_index=True)

temp.head()
mvp_cand.head(10)


#######
# EXRACTING TEAM STATS
# team_standings : Teams stats year by year from 2004
team_standings = pd.DataFrame()
for i in tqdm(range(2004, year)):
    team_east_table = pd.read_html(f"https://www.basketball-reference.com/leagues/NBA_{i}_standings.html", header=0)[0]
    team_east_table = team_east_table.rename({"Eastern Conference" : "team"}, axis=1)
    team_east_table = team_east_table[team_east_table['team'].str.contains('Division') == False]
    team_east_table["seed"] = team_east_table["W"].rank(ascending=False)

    team_west_table = pd.read_html(f"https://www.basketball-reference.com/leagues/NBA_{i}_standings.html", header=0)[1]
    team_west_table = team_west_table.rename({"Western Conference": "team"}, axis=1)
    team_west_table = team_west_table[team_west_table['team'].str.contains('Division') == False]
    team_west_table["seed"] = team_west_table["W"].rank(ascending=False)

    temp = pd.concat([team_east_table, team_west_table], ignore_index=True)
    temp["team"] = temp["team"].str.replace("*", "")
    temp["year"] = i
    team_standings = team_standings.append(temp)

##### Getting Team Abbreviations
team_standings["team"] = team_standings["team"].apply(fix_team_names)
all_teams = pd.read_csv("data/base/all_teams.csv")
team_standings["Tm"] = team_standings.merge(all_teams, how="left", left_on="team", right_on="full_name").get(["abbreviation"])


team_standings.shape
mvp_cand.Player.head()
team_standings.head(20)
team_standings.tail(20)
##### ADVANCE STATS
for name in mvp_cand.Player:
    players.get_stats("Tim Duncan", stat_type="Advanced")



