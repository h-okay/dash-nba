import numpy as np
import pandas as pd
from tqdm import tqdm
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players
from nba_api.stats.static import teams
from time import sleep
import pickle as pkl

pd.set_option("display.max_columns", None)


def get_data():

    # Teams
    print("Updating teams...")
    nba_teams = teams.get_teams()
    all_teams = pd.DataFrame(nba_teams)
    all_teams.to_csv("../data/all_teams.csv", index=False)

    # -------------------------------------------------------------------------

    # All Players
    nba_players = players.get_players()
    all_players = pd.DataFrame(nba_players)
    all_players.to_csv("../data/all_players.csv", index=False)

    # -------------------------------------------------------------------------

    # Active Players
    print("Updating players...")
    nba_players_ = players.get_active_players()
    active_players = pd.DataFrame(nba_players_)

    # -------------------------------------------------------------------------

    # Matches
    team_ids = all_teams["id"].to_list()
    team_ids = list(map(lambda x: str(x), team_ids))
    matches = {}
    for id in tqdm(team_ids, desc="Updating matches"):
        gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=id)
        team_name = list(
            set(gamefinder.get_data_frames()[0]["TEAM_NAME"].to_list()))[0]

        # Name Changes
        if team_name == "New Jersey Nets":
            team_name = "Brooklyn Nets"
        if team_name == "Washington Bullets":
            team_name = "Washington Wizards"
        if team_name in ["LA Clippers", "San Diego Clippers"]:
            team_name = "Los Angeles Clippers"
        if team_name == "Kansas City Kings":
            team_name = "Sacramento Kings"
        if team_name == "Seattle SuperSonics":
            team_name = "Oklahoma City Thunder"
        if team_name in [
                "New Orleans/Oklahoma City Hornets", "New Orleans Hornets"
        ]:
            team_name = "New Orleans Pelicans"
        if team_name == "Charlotte Bobcats":
            team_name = "Charlotte Hornets"
        if team_name == "Vancouver Grizzlies":
            team_name = "Memphis Grizzlies"

        matches[team_name] = gamefinder.get_data_frames()[0]
        matches[team_name].SEASON_ID = matches[team_name].SEASON_ID.apply(
            lambda x: (x[1:] + f"-{int(x[-2:]) + 1}") if x[2:4] != "00" or x ==
            "22009" else (x[1:] + f"-0{int(x[-2:]) + 1}"))
        matches[team_name].SEASON_ID = matches[team_name].SEASON_ID.apply(
            lambda x: "1999-00" if x == "1999-100" else x)
        sleep(0.600)

    for key in matches.keys():
        temp = matches[key]
        temp["date"] = pd.to_datetime(temp.SEASON_ID.apply(lambda x: x[:5]))
        temp = temp[temp["date"] >= "1983-01-01"]
        temp = temp.drop("date", axis=1)
        matches[key] = temp

    with open("../data/matches.pkl", "wb") as file:
        pkl.dump(matches, file)

    # -------------------------------------------------------------------------

    # Stats
    player_ids = active_players["id"].to_list()
    player_ids = list(map(lambda x: str(x), player_ids))
    stats = pd.DataFrame()
    for id in tqdm(player_ids, desc="Updating player stats"):
        career = playercareerstats.PlayerCareerStats(player_id=id)
        p_st = career.get_data_frames()[0]
        p_st = p_st[p_st.SEASON_ID == "2021-22"]
        stats = stats.append(p_st, ignore_index=True)
        sleep(0.600)

    all_stats = pd.read_csv("../data/fixed_raw_stats.csv")
    all_stats = all_stats[all_stats.SEASON_ID >= "1983-84"]
    all_stats.drop(all_stats[all_stats.SEASON_ID == "2021-22"].index,
                   axis=0,
                   inplace=True)
    updated_stats = pd.concat([all_stats, stats], axis=0, ignore_index=True)
    updated_stats.to_csv("../data/stats.csv", index=False)

    # -------------------------------------------------------------------------

    # Merge Final
    merged = all_players.merge(updated_stats,
                               left_on="id",
                               right_on="PLAYER_ID",
                               how="left").merge(all_teams,
                                                 left_on="TEAM_ID",
                                                 right_on="id",
                                                 how="left")

    merged.drop(
        [
            "id_x",
            "is_active",
            "LEAGUE_ID",
            "full_name_x",
            "id_y",
            "abbreviation",
            "nickname",
            "city",
            "state",
            "year_founded",
        ],
        axis=1,
        inplace=True,
    )
    merged = merged.rename(columns=({
        "PLAYER_ID": "P_ID",
        "PLAYER_AGE": "AGE",
        "full_name_y": "TEAM",
        "first_name": "FIRST_NAME",
        "last_name": "LAST_NAME",
        "FG_PCT": "FG%",
        "FG3_PCT": "FG3%",
    }))
    merged.dropna(axis=0, inplace=True)
    merged.AGE = merged.AGE.astype("int64")
    merged.MIN = merged.MIN.astype("int64")
    merged.GP = merged.GP.astype("int64")
    min_gp = np.round(
        pd.pivot_table(merged, values="MIN", columns="SEASON_ID",
                       index="P_ID") /
        pd.pivot_table(merged, values="GP", columns="SEASON_ID", index="P_ID"),
        2,
    )

    min_gp = min_gp.unstack().to_frame().reset_index()
    min_gp.columns = ["SEASON_ID", "P_ID", "MPG"]
    merged = merged.merge(min_gp, on=["SEASON_ID", "P_ID"])
    merged = merged[merged["MPG"] >= 6.09]
    merged = merged[merged["MIN"] >= 500]

    merged.to_csv("../data/merged.csv", index=False)
    print("Updating done.")
