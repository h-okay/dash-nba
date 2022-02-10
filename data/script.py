import pandas as pd
from pandas.core.common import SettingWithCopyWarning
from data.scripts.classes import (playerRating, Generators, getStandings,
                                  playoffWins, Schedule, PER)
from data.scripts.api import get_data
from data.scripts.performanceForecast import get_player_perf_forecast
from data.scripts.elo import calculate_elo
import warnings

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

if __name__ == "__main__":

    while True:
        update = (input(
            "WARNING: Daily updates can take up to 20 minutes. \nUpdate data? [y/n]\n"
        ).strip().lower())
        if update == "y":
            print("Updating data, please wait... ")
            gsu = getStandings()
            gsu.update()
            get_data()
            print("Done.")
            break
        elif update == "n":
            break
        continue

    merged = pd.read_csv("../data/base/merged.csv")
    available_seasons = sorted(merged.SEASON_ID.unique())[::-1]

    glob = pd.DataFrame()
    for season in available_seasons:
        # PER
        print('-' * 37)
        print(f"Generating data for {season} season...")
        df = playerRating(season)
        df.factor()
        df.vop()
        df.drbp()
        df.add_uper()
        df.team_pace()
        df.league_pace()
        df.ratings()
        df.merged.to_csv(f'../data/base/pers/PER_{season}.csv', index=False)
        df.team_pers()
        first = df.team_stats

        #Binaries
        bg = Generators()
        bg.playoff_generator()
        second = bg.champ_generator()
        second = second[second.SEASON_ID == season][[
            "TEAM", "SEASON_ID", "Playoff", "Quarter", "Semi", "Final",
            "Champion"
        ]]

        # Merge
        final = first.merge(second,
                            left_on=["TEAM", "SEASON"],
                            right_on=["TEAM", "SEASON_ID"],
                            how="left")
        final.drop(["SEASON_ID"], axis=1, inplace=True)
        glob = pd.concat([glob, final], axis=0)
        print('-' * 37 + '\n')

    # Playoff Win Count
    print('Getting playoffs...')
    pw = playoffWins()
    third = pw.add_playoff_wins()
    glob = glob.merge(third, on=["TEAM", "SEASON"], how="left")

    # Standings
    print('Getting standings...')
    gs = getStandings()
    fourth = gs.all_standings()
    glob = glob.merge(fourth, on=["TEAM", "SEASON"], how="left")

    # ELO
    print('Getting ELOs...')
    calculate_elo()
    all_teams = pd.read_csv('../data/base/all_teams.csv')
    elos = pd.read_csv('../data/base/elos.csv')
    elos = elos.merge(all_teams, left_on='TEAM_ID', right_on='id', how='left')
    elos.drop([
        'id', 'nickname', 'city', 'state', 'year_founded', 'DATE',
        'abbreviation'
    ],
              axis=1,
              inplace=True)
    elos = elos.rename(columns={'full_name': 'TEAM'})
    glob = glob.merge(elos, on=['SEASON', 'TEAM'], how='left')

    # Schedule
    print('Getting schedules...')
    schedule = Schedule()
    schedule.get_schedules()

    #PER
    print('Getting PER DataFrame...')
    per = PER()

    # PER Forecast
    get_player_perf_forecast()

    # DUMP
    print('Generating ML Ready DataFrame...')
    glob.to_csv("../data/est/mlready.csv", index=False)
    print("Done")
