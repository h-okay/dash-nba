import pickle as pkl
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy import spatial
from scipy.stats import weightedtau

from time import sleep
import datetime
import glob

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service

from data.scripts.helpers import fix_team_names, get_names


class playerRating:

    def __init__(self, season):
        self.season = season
        self.all_players = pd.read_csv("../data/base/all_players.csv")
        self.all_teams = pd.read_csv("../data/base/all_teams.csv")
        self.merged = pd.read_csv("../data/base/merged.csv")
        self.merged = self.merged[self.merged["SEASON_ID"] == self.season]
        self.team_stats = self.merged.groupby("TEAM")[[
            "FGM",
            "FGA",
            "FG3M",
            "FG3A",
            "FTM",
            "FTA",
            "OREB",
            "DREB",
            "REB",
            "AST",
            "STL",
            "BLK",
            "TOV",
            "PF",
            "PTS",
        ]].sum()
        with open("../data/base/matches.pkl", "rb") as file:
            self.matches = pkl.load(file)

    def factor(self):
        lg_AST = self.merged["AST"].sum()
        lg_FG = self.merged["FGM"].sum()
        lg_FT = self.merged["FTM"].sum()
        self.merged["factor"] = ((2 / 3) * (0.5 * (lg_AST) / lg_FG) /
                                 (2 * (lg_FG / lg_FT)))
        return self.merged

    def vop(self):
        lg_PTS = self.merged["PTS"].sum()
        lg_FGA = self.merged["FGA"].sum()
        lg_ORB = self.merged["OREB"].sum()
        lg_TOV = self.merged["TOV"].sum()
        lg_FTA = self.merged["FTA"].sum()
        self.merged["vop"] = lg_PTS / (lg_FGA - lg_ORB + lg_TOV +
                                       0.44 * lg_FTA)
        return self.merged

    def drbp(self):
        lg_TRB = self.merged["REB"].sum()
        lg_ORB = self.merged["OREB"].sum()
        self.merged["drbp"] = (lg_TRB - lg_ORB) / lg_TRB
        return self.merged

    def uPER(self, team):
        MP = self.merged["MIN"]
        FG3M = self.merged["FG3M"]
        AST = self.merged["AST"]
        factor = self.merged["factor"]
        FG = self.merged["FGM"]
        FT = self.merged["FTM"]
        VOP = self.merged["vop"]
        TOV = self.merged["TOV"]
        DRBP = self.merged["drbp"]
        FGA = self.merged["FGA"]
        FTA = self.merged["FTA"]
        TRB = self.merged["REB"]
        ORB = self.merged["OREB"]
        STL = self.merged["STL"]
        BLK = self.merged["BLK"]
        PF = self.merged["PF"]
        lg_FT = self.merged["FTM"].sum()
        lg_PF = self.merged["PF"].sum()
        lg_FTA = self.merged["FTA"].sum()
        mask = self.team_stats[self.team_stats.index == team]
        team_AST = mask["AST"].values[0]
        team_FG = mask["FGM"].values[0]
        uPER = (1 / MP) * (FG3M + (2 / 3) * AST + (2 - factor *
                                                   (team_AST / team_FG)) * FG +
                           (FT * 0.5 *
                            (1 + (1 - (team_AST / team_FG)) + (2 / 3) *
                             (team_AST / team_FG))) - VOP * TOV - VOP * DRBP *
                           (FGA - FG) - VOP * 0.44 * (0.44 + (0.56 * DRBP)) *
                           (FTA - FT) + VOP * (1 - DRBP) *
                           (TRB - ORB) + VOP * DRBP * ORB + VOP * STL +
                           VOP * DRBP * BLK - PF * ((lg_FT / lg_PF) - 0.44 *
                                                    (lg_FTA / lg_PF) * VOP))
        self.merged.loc[self.merged.TEAM == team, "uPER"] = uPER
        return self.merged

    def add_uper(self):
        for team in self.merged.TEAM.unique():
            self.uPER(team)
        return self.merged

    def generate_played(self, team):
        df = self.matches[team]
        df = df[df["SEASON_ID"] == self.season]
        df["OPPONENT"] = df["MATCHUP"].apply(lambda x: x[-3:])
        df = df.merge(self.all_teams,
                      left_on="OPPONENT",
                      right_on="abbreviation")
        played = [(team, opp) for opp in df["full_name"].values]
        return played

    def get_team_paces(self, team, opponent):
        # TEAM
        team = self.team_stats[self.team_stats.index == team]
        team_AST = team["AST"].values[0]
        team_FG = team["FGM"].values[0]
        team_FGA = team["FGA"].values[0]
        team_FTA = team["FTA"].values[0]
        team_ORB = team["OREB"].values[0]
        team_TOV = team["TOV"].values[0]
        team_DRB = team["DREB"].values[0]
        # OPPONENT
        opponent = self.team_stats[self.team_stats.index == opponent]
        opp_AST = opponent["AST"].values[0]
        opp_FG = opponent["FGM"].values[0]
        opp_FGA = opponent["FGA"].values[0]
        opp_FTA = opponent["FTA"].values[0]
        opp_ORB = opponent["OREB"].values[0]
        opp_TOV = opponent["TOV"].values[0]
        opp_DRB = opponent["DREB"].values[0]
        team_pace = 0.5 * ((team_FGA + 0.4 * team_FTA - 1.07 *
                            (team_ORB / (team_ORB + opp_DRB)) *
                            (team_FGA - team_FG) + team_TOV) +
                           (opp_FGA + 0.4 * opp_FTA - 1.07 *
                            (opp_ORB / opp_ORB + team_DRB)) *
                           (opp_FGA - opp_FG) + opp_TOV)
        return team_pace

    def t_pace(self, team):
        self.merged.loc[self.merged.TEAM == team, "T_PACE"] = np.array([
            self.get_team_paces(matchup[0], matchup[1])
            for matchup in self.generate_played(team)
        ]).mean()
        return self.merged

    def league_pace(self):
        total = 0
        count = 0
        for key in tqdm(self.matches.keys(), desc="Calculating league pace"):
            played = self.generate_played(key)
            count += len(played)
            for game in played:
                team_pace = self.get_team_paces(game[0], game[1])
                total += team_pace
        lg_pace = total / count
        self.merged.loc[self.merged.SEASON_ID == self.season,
                        "L_PACE"] = lg_pace
        return self.merged

    def team_pace(self):
        teams = self.merged.TEAM.unique()
        for team in tqdm(teams, desc="Calculating team pace..."):
            self.merged = self.t_pace(team)
        return self.merged

    def ratings(self):
        self.merged[
            "adjustment"] = self.merged["L_PACE"] / self.merged["T_PACE"]
        self.merged["aPER"] = self.merged["uPER"] * self.merged["adjustment"]
        self.merged["PER"] = np.round(
            self.merged["aPER"] * (15 / self.merged["aPER"].mean()), 2)
        return self.merged

    def team_pers(self):
        tempdf = np.round(self.merged.groupby("TEAM").PER.mean(), 2)
        self.team_stats = self.team_stats.merge(tempdf,
                                                left_on=self.team_stats.index,
                                                right_on=tempdf.index,
                                                how="left")
        self.team_stats = self.team_stats.rename(columns={"key_0": "TEAM"})
        self.team_stats["SEASON"] = self.season

        return self.team_stats


class Generators:

    def __init__(self):
        self.merged = pd.read_csv("../data/base/merged.csv")
        self.merged.columns = [col.strip() for col in self.merged.columns]
        self.playoffs = pd.read_csv("../data/base/playoffs.csv")
        self.playoffs.columns = [col.strip() for col in self.playoffs.columns]
        # self.playoffs.Year = self.playoffs.Year.apply(
        #     lambda x: str(x - 2) + "-" + str(int(str(x-2)[-2:])+1 if x != '2001'
        #                                      else "1999-00"))
        self.playoffs.Year = self.playoffs.Year.apply(
            lambda x: str(x - 1) + "-" + str(x)[-2:])
        self.playoffs.columns = [
            col.strip().upper() for col in self.playoffs.columns
        ]
        self.playoffs.SERIES = self.playoffs.SERIES.apply(lambda x: x.strip())
        self.t_stats = (self.merged.groupby(["SEASON_ID", "TEAM"])[[
            "FGM",
            "FGA",
            "FG3M",
            "FG3A",
            "FTM",
            "FTA",
            "OREB",
            "DREB",
            "REB",
            "AST",
            "STL",
            "BLK",
            "TOV",
            "PF",
            "PTS",
        ]].sum().reset_index())

        self.t_stats = self.t_stats[self.t_stats.SEASON_ID >= "1979-80"]
        self.seasons = [
            i for i in self.t_stats.SEASON_ID.unique() if i != "2021-22"
        ]

    def playoff_generator(self):
        for season in self.seasons:
            temp1 = self.playoffs[self.playoffs.YEAR == season]
            for stage in ["Playoff", "Quarter", "Semi", "Final"]:
                temp2 = temp1[temp1.SERIES == stage]
                a = temp2[["YEAR", "TEAM1"]].rename(columns={"TEAM1": "TEAM"})
                b = temp2[["YEAR", "TEAM2"]].rename(columns={"TEAM2": "TEAM"})
                c = pd.concat([a, b], axis=0, ignore_index=True)
                d = [i.strip() for i in c.TEAM.to_list()]
                self.t_stats.loc[(self.t_stats.SEASON_ID == season) &
                                 (self.t_stats.TEAM.isin(d)), stage, ] = 1
                self.t_stats[stage] = self.t_stats[stage].fillna(0)
                self.t_stats[stage] = self.t_stats[stage].astype("int64")
        return self.t_stats

    def champ_generator(self):
        for season in self.seasons:
            chmp = self.playoffs.loc[(self.playoffs.YEAR == season) &
                                     (self.playoffs.SERIES == "Final"),
                                     "WINNER", ].values[0]
            self.t_stats.loc[(self.t_stats.SEASON_ID == season) &
                             (self.t_stats.TEAM == chmp), "Champion", ] = 1
            self.t_stats["Champion"] = self.t_stats["Champion"].fillna(0)
            self.t_stats["Champion"] = self.t_stats["Champion"].astype("int64")
        return self.t_stats


class getStandings:

    def __init__(self):
        self.years = range(1971, 2023)
        self.seasons = [
            str(year - 1) + "-" + str(year)[-2:] for year in self.years
        ]
        self.numeric = pd.DataFrame({
            "year": self.years,
            "season": self.seasons
        })
        self.standings = pd.read_csv('../data/base/standingsCleaned.csv')
        self.standings.columns = [
            col.strip() for col in self.standings.columns
        ]
        self.standings.TEAM = self.standings.TEAM.apply(lambda x: x.strip())

    def current_standings(self):
        self.current = self.standings[self.standings.SEASON == '2021-22']
        return self.current

    def all_standings(self):
        return self.standings

    def update(self):
        self.standings = self.standings[self.standings.SEASON != '2021-22']
        timeout = 5
        ser = Service("C:/Program Files/chromedriver.exe")
        s = webdriver.Chrome(service=ser)
        url = "https://www.nba.com/standings?GroupBy=conf&Season=2021-22&Section=overall"
        s.get(url)
        element_present = EC.presence_of_element_located((By.CLASS_NAME, 'h5'))
        WebDriverWait(s, timeout).until(element_present)
        html = s.page_source
        tables = pd.read_html(html)
        data1 = tables[0]
        data2 = tables[1]
        data = pd.concat([data1, data2], axis=0, ignore_index=True)
        s.quit()
        data.TEAM = data.TEAM.str.extract("([A-Za-z].*)")
        data.TEAM = data.TEAM.apply(lambda x: x[:-4])
        data['SEASON'] = '2021-22'
        self.standings = pd.concat([self.standings, data],
                                   axis=0,
                                   ignore_index=True)
        self.standings.TEAM = self.standings.TEAM.apply(get_names)
        self.standings.TEAM = self.standings.TEAM.apply(fix_team_names)
        self.standings.to_csv('../data/base/standingsCleaned.csv', index=False)


class playoffWins:

    def __init__(self):
        self.playoffwins = pd.read_csv("../data/base/playoffwins.csv")
        self.playoffwins.columns = [
            col.strip() for col in self.playoffwins.columns
        ]
        self.playoffwins.TEAM = self.playoffwins.TEAM.apply(fix_team_names)

    def add_playoff_wins(self):
        return self.playoffwins


class Schedule:
    def __init__(self):
        self.today = datetime.datetime.today().strftime('%Y-%m-%d')
        self.dates = pd.date_range(start=self.today, end='2022-04-10',
                              freq='D').to_frame().reset_index(drop=True)
        self.dates.columns = ['date']
        self.dates['date'] = self.dates['date'].apply(lambda x: x.strftime('%Y%m%d'))
        self.dates = self.dates.date.to_list()

    def get_schedules(self):
        self.schedule = pd.DataFrame()
        for date in self.dates:
            try:
                df = pd.read_html(
                    f'https://www.cbssports.com/nba/schedule/{date}/')[0]
                df = df[['Away', 'Home', 'Time / TV']]
                df['date'] = date
                self.schedule = pd.concat([self.schedule, df])
            except ValueError:
                continue
        self.schedule.date = pd.to_datetime(self.schedule.date)
        self.schedule = self.schedule.sort_values(by='date')
        self.schedule.Away = self.schedule.Away.apply(fix_team_names)
        self.schedule.Home = self.schedule.Home.apply(fix_team_names)
        self.schedule.to_csv('../data/base/schedule.csv', index=False)


class PER:
    def __init__(self):
        self.all_filenames = [file for file in glob.glob("C:/Users/Hakan/Desktop/GitHub/VBO/data/base/pers" + "./*.csv")]
        self.all_df = pd.concat([pd.read_csv(f) for f in self.all_filenames])
        self.all_df.to_csv(
            "C:/Users/Hakan/Desktop/GitHub/VBO/data/base/" + "per.csv", index=False)


# class Metrics:
#
#     def __init__(self, year):
#         self.season = str(year - 1) + "-" + str(year)[-2:]
#
#     def compare_similarity(self, num):
#         real = pd.read_csv(
#             f"real/realper{self.season}.csv").iloc[:num, :].PER.to_list()
#         est = pd.read_csv(
#             f"estimations/{self.season}.csv").iloc[:num, :].PER.to_list()
#         result = 1 - spatial.distance.cosine(real, est)
#         tau, pvalue = weightedtau(real, est)
#         return result, tau
#
#     def similarity_score(self):
#         metrics = pd.DataFrame(columns=["Cosine", "Tau"])
#         for i in [10, 25, 50, 100, 200]:
#             result, tau = self.compare_similarity(i)
#             temp_df = pd.DataFrame({
#                 "Cosine": str(result),
#                 "Tau": str(tau)
#             },
#                                    index=[f"First {i}"])
#             metrics = metrics.append([temp_df])
#         return metrics
