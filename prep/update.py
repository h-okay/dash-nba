import pandas as pd
from pandas.core.common import SettingWithCopyWarning
from prep.scripts.classes import (
    playerRating,
    Generators,
    getStandings,
    playoffWins,
    Schedule,
    PER,
    ELO,
    PERForecast,
    MVPForecast,
    print_done,
)
from prep.scripts.api import get_data
from tqdm import tqdm

import warnings

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

if __name__ == "__main__":

    while True:
        update = (
            input(
                "WARNING: Daily updates can take up to 20 minutes. \nUpdate data? [y/n]\n"
            )
            .strip()
            .lower()
        )
        if update == "y":
            print_done("Updating")
            # Matches, Players ---
            get_data()
            # Standings ---
            gsu = getStandings()
            gsu.update()
            # MVP ---
            mvp = MVPForecast()
            mvp.prep_and_predict()
            print("[DONE]")
            break
        elif update == "n":
            break
        continue

    merged = pd.read_csv("../prep/data/merged.csv")
    available_seasons = sorted(merged.SEASON_ID.unique())[::-1]

    glob = pd.DataFrame()
    pbar = tqdm(
        available_seasons, desc="Generating seasonal PERs...", position=0, leave=False
    )
    for season in pbar:
        # PER
        pbar.set_description(f"{season}")
        df = playerRating(season)
        df.factor()
        df.vop()
        df.drbp()
        df.add_uper()
        df.team_pace()
        df.league_pace()
        df.ratings()
        df.merged.to_csv(f"../prep/data/pers/PER_{season}.csv", index=False)
        df.team_pers()
        first = df.team_stats

        # Binaries
        bg = Generators()
        bg.playoff_generator()
        second = bg.champ_generator()
        second = second[second.SEASON_ID == season][
            ["TEAM", "SEASON_ID", "Playoff", "Quarter", "Semi", "Final", "Champion"]
        ]

        # Merge
        final = first.merge(
            second,
            left_on=["TEAM", "SEASON"],
            right_on=["TEAM", "SEASON_ID"],
            how="left",
        )
        final.drop(["SEASON_ID"], axis=1, inplace=True)
        glob = pd.concat([glob, final], axis=0)
    print("Generating seasonal PERs...    [DONE]")

    # Playoff Win Count
    print_done("Getting playoffs")
    pw = playoffWins()
    third = pw.add_playoff_wins()
    glob = glob.merge(third, on=["TEAM", "SEASON"], how="left")
    print("[DONE]")

    # Standings
    print_done("Getting standings")
    gs = getStandings()
    fourth = gs.all_standings()
    glob = glob.merge(fourth, on=["TEAM", "SEASON"], how="left")
    print("[DONE]")

    # ELO
    print_done("Generating ELOS")
    elo = ELO()
    elo.calculate_elo()
    all_teams = pd.read_csv("../prep/data/all_teams.csv")
    elos = pd.read_csv("../prep/data/elos.csv")
    elos = elos.merge(all_teams, left_on="TEAM_ID", right_on="id", how="left")
    elos.drop(
        ["id", "nickname", "city", "state", "year_founded", "DATE", "abbreviation"],
        axis=1,
        inplace=True,
    )
    elos = elos.rename(columns={"full_name": "TEAM"})
    glob = glob.merge(elos, on=["SEASON", "TEAM"], how="left")
    print("Generating ELOS...     [DONE]")

    # Schedule
    print_done("Getting Schedules")
    schedule = Schedule()
    schedule.get_schedules()
    print("[DONE]")

    # PER
    print_done("Getting PER DataFrame")
    per = PER()
    print("[DONE]")

    # PER Forecast
    print_done("Generating PER Predictions")
    perfor = PERForecast()
    perfor.get_player_perf_forecast()
    print("[DONE]")

    # DUMP
    print_done("Generating ML Ready DataFrame")
    glob.to_csv("../prep/data/mlready.csv", index=False)
    print("[DONE]")
