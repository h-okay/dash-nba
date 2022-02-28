import pickle as pkl
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import sleep
import datetime
import glob
import catboost

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service

from utils.helpers import fix_team_names, get_names, FastML

pd.options.mode.chained_assignment = None


def print_done(category):
    print(f"{category}...      ", end="", flush=True)


class playerRating:
    def __init__(self, season):
        self.season = season
        self.all_players = pd.read_csv("../prep/data/all_players.csv")
        self.all_teams = pd.read_csv("../prep/data/all_teams.csv")
        self.merged = pd.read_csv("../prep/data/merged.csv")
        self.merged = self.merged[self.merged["SEASON_ID"] == self.season]
        self.team_stats = self.merged.groupby("TEAM")[
            [
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
            ]
        ].sum()
        with open("../prep/data/matches.pkl", "rb") as file:
            self.matches = pkl.load(file)

    def factor(self):
        lg_AST = self.merged["AST"].sum()
        lg_FG = self.merged["FGM"].sum()
        lg_FT = self.merged["FTM"].sum()
        self.merged["factor"] = (
            (2 / 3) * (0.5 * (lg_AST) / lg_FG) / (2 * (lg_FG / lg_FT))
        )
        return self.merged

    def vop(self):
        lg_PTS = self.merged["PTS"].sum()
        lg_FGA = self.merged["FGA"].sum()
        lg_ORB = self.merged["OREB"].sum()
        lg_TOV = self.merged["TOV"].sum()
        lg_FTA = self.merged["FTA"].sum()
        self.merged["vop"] = lg_PTS / (lg_FGA - lg_ORB + lg_TOV + 0.44 * lg_FTA)
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
        uPER = (1 / MP) * (
            FG3M
            + (2 / 3) * AST
            + (2 - factor * (team_AST / team_FG)) * FG
            + (
                FT
                * 0.5
                * (1 + (1 - (team_AST / team_FG)) + (2 / 3) * (team_AST / team_FG))
            )
            - VOP * TOV
            - VOP * DRBP * (FGA - FG)
            - VOP * 0.44 * (0.44 + (0.56 * DRBP)) * (FTA - FT)
            + VOP * (1 - DRBP) * (TRB - ORB)
            + VOP * DRBP * ORB
            + VOP * STL
            + VOP * DRBP * BLK
            - PF * ((lg_FT / lg_PF) - 0.44 * (lg_FTA / lg_PF) * VOP)
        )
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
        df = df.merge(self.all_teams, left_on="OPPONENT", right_on="abbreviation")
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
        team_pace = 0.5 * (
            (
                team_FGA
                + 0.4 * team_FTA
                - 1.07 * (team_ORB / (team_ORB + opp_DRB)) * (team_FGA - team_FG)
                + team_TOV
            )
            + (opp_FGA + 0.4 * opp_FTA - 1.07 * (opp_ORB / opp_ORB + team_DRB))
            * (opp_FGA - opp_FG)
            + opp_TOV
        )
        return team_pace

    def t_pace(self, team):
        self.merged.loc[self.merged.TEAM == team, "T_PACE"] = np.array(
            [
                self.get_team_paces(matchup[0], matchup[1])
                for matchup in self.generate_played(team)
            ]
        ).mean()
        return self.merged

    def league_pace(self):
        total = 0
        count = 0
        for key in self.matches.keys():
            played = self.generate_played(key)
            count += len(played)
            for game in played:
                team_pace = self.get_team_paces(game[0], game[1])
                total += team_pace
        lg_pace = total / count
        self.merged.loc[self.merged.SEASON_ID == self.season, "L_PACE"] = lg_pace
        return self.merged

    def team_pace(self):
        teams = self.merged.TEAM.unique()
        for team in teams:
            self.merged = self.t_pace(team)
        return self.merged

    def ratings(self):
        self.merged["adjustment"] = self.merged["L_PACE"] / self.merged["T_PACE"]
        self.merged["aPER"] = self.merged["uPER"] * self.merged["adjustment"]
        self.merged["PER"] = np.round(
            self.merged["aPER"] * (15 / self.merged["aPER"].mean()), 2
        )
        self.merged.drop(
            ["adjustment", "L_PACE", "T_PACE", "aPER", "uPER", "factor", "vop", "drbp"],
            axis=1,
            inplace=True,
        )
        return self.merged

    def team_pers(self):
        tempdf = np.round(self.merged.groupby("TEAM").PER.mean(), 2)
        self.team_stats = self.team_stats.merge(
            tempdf, left_on=self.team_stats.index, right_on=tempdf.index, how="left"
        )
        self.team_stats = self.team_stats.rename(columns={"key_0": "TEAM"})
        self.team_stats["SEASON"] = self.season

        return self.team_stats


class Generators:
    def __init__(self):
        self.merged = pd.read_csv("../prep/data/merged.csv")
        self.merged.columns = [col.strip() for col in self.merged.columns]
        self.playoffs = pd.read_csv("../prep/data/playoffs.csv")
        self.playoffs.columns = [col.strip() for col in self.playoffs.columns]
        self.playoffs.Year = self.playoffs.Year.apply(
            lambda x: str(x - 1) + "-" + str(x)[-2:]
        )
        self.playoffs.columns = [col.strip().upper() for col in self.playoffs.columns]
        self.playoffs.SERIES = self.playoffs.SERIES.apply(lambda x: x.strip())
        self.t_stats = (
            self.merged.groupby(["SEASON_ID", "TEAM"])[
                [
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
                ]
            ]
            .sum()
            .reset_index()
        )

        self.t_stats = self.t_stats[self.t_stats.SEASON_ID >= "1979-80"]
        self.seasons = [i for i in self.t_stats.SEASON_ID.unique() if i != "2021-22"]

    def playoff_generator(self):
        for season in self.seasons:
            temp1 = self.playoffs[self.playoffs.YEAR == season]
            for stage in ["Playoff", "Quarter", "Semi", "Final"]:
                temp2 = temp1[temp1.SERIES == stage]
                a = temp2[["YEAR", "TEAM1"]].rename(columns={"TEAM1": "TEAM"})
                b = temp2[["YEAR", "TEAM2"]].rename(columns={"TEAM2": "TEAM"})
                c = pd.concat([a, b], axis=0, ignore_index=True)
                d = [i.strip() for i in c.TEAM.to_list()]
                self.t_stats.loc[
                    (self.t_stats.SEASON_ID == season) & (self.t_stats.TEAM.isin(d)),
                    stage,
                ] = 1
                self.t_stats[stage] = self.t_stats[stage].fillna(0)
                self.t_stats[stage] = self.t_stats[stage].astype("int64")
        return self.t_stats

    def champ_generator(self):
        for season in self.seasons:
            chmp = self.playoffs.loc[
                (self.playoffs.YEAR == season) & (self.playoffs.SERIES == "Final"),
                "WINNER",
            ].values[0]
            self.t_stats.loc[
                (self.t_stats.SEASON_ID == season) & (self.t_stats.TEAM == chmp),
                "Champion",
            ] = 1
            self.t_stats["Champion"] = self.t_stats["Champion"].fillna(0)
            self.t_stats["Champion"] = self.t_stats["Champion"].astype("int64")
        return self.t_stats


class getStandings:
    def __init__(self):
        self.years = range(1971, 2023)
        self.seasons = [str(year - 1) + "-" + str(year)[-2:] for year in self.years]
        self.numeric = pd.DataFrame({"year": self.years, "season": self.seasons})
        self.standings = pd.read_csv("../prep/data/standingsCleaned.csv")
        self.standings.columns = [col.strip() for col in self.standings.columns]
        self.standings.TEAM = self.standings.TEAM.apply(lambda x: x.strip())

    def current_standings(self):
        self.current = self.standings[self.standings.SEASON == "2021-22"]
        return self.current

    def all_standings(self):
        return self.standings

    def update(self):
        self.standings = self.standings[self.standings.SEASON != "2021-22"]
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36"
        timeout = 5
        ser = Service("C:/Program Files/chromedriver.exe")
        s = webdriver.Chrome(service=ser)
        url = (
            "https://www.nba.com/standings?GroupBy=conf&Season=2021-22&Section=overall"
        )
        s.get(url)
        element_present = EC.presence_of_element_located((By.CLASS_NAME, "h5"))
        WebDriverWait(s, timeout).until(element_present)
        html = s.page_source
        tables = pd.read_html(html)
        data1 = tables[0]
        data2 = tables[1]
        data = pd.concat([data1, data2], axis=0, ignore_index=True)
        s.quit()
        data.TEAM = data.TEAM.str.extract("([A-Za-z].*)")
        data.TEAM = data.TEAM.apply(lambda x: x[:-4])
        data["SEASON"] = "2021-22"
        self.standings = pd.concat([self.standings, data], axis=0, ignore_index=True)
        self.standings.TEAM = self.standings.TEAM.apply(get_names)
        self.standings.TEAM = self.standings.TEAM.apply(fix_team_names)
        self.standings.to_csv("../prep/data/standingsCleaned.csv", index=False)


class playoffWins:
    def __init__(self):
        self.playoffwins = pd.read_csv("../prep/data/playoffwins.csv")
        self.playoffwins.columns = [col.strip() for col in self.playoffwins.columns]
        self.playoffwins.TEAM = self.playoffwins.TEAM.apply(fix_team_names)

    def add_playoff_wins(self):
        return self.playoffwins


class Schedule:
    def __init__(self):
        self.today = datetime.datetime.today().strftime("%Y-%m-%d")
        self.dates = (
            pd.date_range(start=self.today, end="2022-04-10", freq="D")
            .to_frame()
            .reset_index(drop=True)
        )
        self.dates.columns = ["date"]
        self.dates["date"] = self.dates["date"].apply(lambda x: x.strftime("%Y%m%d"))
        self.dates = self.dates.date.to_list()

    def get_schedules(self):
        self.schedule = pd.DataFrame()
        for date in self.dates:
            try:
                df = pd.read_html(f"https://www.cbssports.com/nba/schedule/{date}/")[0]
                df = df[["Away", "Home", "Time / TV"]]
                df["date"] = date
                self.schedule = pd.concat([self.schedule, df])
            except ValueError:
                continue
        self.schedule.date = pd.to_datetime(self.schedule.date)
        self.schedule = self.schedule.sort_values(by="date")
        self.schedule.Away = self.schedule.Away.apply(fix_team_names)
        self.schedule.Home = self.schedule.Home.apply(fix_team_names)
        self.schedule.to_csv("../prep/data/schedule.csv", index=False)


class PER:
    def __init__(self):
        self.all_filenames = [
            file for file in glob.glob("../prep/data/pers" + "./*.csv")
        ]
        self.all_df = pd.concat([pd.read_csv(f) for f in self.all_filenames])
        self.all_df.to_csv("../prep/data/" + "per.csv", index=False)


class ELO:
    def __init__(self):
        with open("../prep/data/matches.pkl", "rb") as file:
            self.matches = pkl.load(file)
        for i in self.matches.values():
            cols = i.columns
            break
        self.starter = pd.DataFrame(columns=cols)
        for value in self.matches.values():
            self.starter = pd.concat([self.starter, value]).copy()
        self.starter["AWAY"] = (
            self.starter.apply(lambda x: "@" in x["MATCHUP"], axis=1) * 1
        )
        away_matches = self.starter[self.starter["AWAY"] == 1].copy()
        home_matches = self.starter[self.starter["AWAY"] == 0].copy()
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
        self.elo_ratings = np.full(
            len(team_ids), 1500
        )  # Takımların başlangıç elolarını belirt.
        a = concat_matches[["TEAM_ID_x", "GAME_DATE_x"]]
        self.checkpoint = concat_matches[~a.duplicated()]  # 6 maç drop oldu.
        self.checkpoint = (
            self.checkpoint.sort_values("GAME_DATE_x")
            .reset_index()
            .drop("index", axis=1)
        )

        self.elo_dict = dict(zip(team_ids, self.elo_ratings))
        self.elo_date_team = pd.DataFrame(columns=["DATE", "SEASON", "TEAM_ID", "ELO"])

    @staticmethod
    def expected_score(elo_a, elo_b):
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    @staticmethod
    def new_rating(elo_a, score_a, expected_score_a):
        return elo_a + 32 * (score_a - expected_score_a)

    def calculate_elo(self):
        for index, row in tqdm(
            self.checkpoint.iterrows(),
            total=self.checkpoint.shape[0],
            position=0,
            leave=False,
        ):
            season_id = row["SEASON_ID_x"]
            game_date = row["GAME_DATE_x"]
            home_id = row["TEAM_ID_x"]
            home_elo = self.elo_dict[home_id]
            away_id = row["TEAM_ID_y"]
            away_elo = self.elo_dict[away_id]
            exp_scr = self.expected_score(home_elo, away_elo)
            result = row["WL_x"]

            if result == "W":
                self.elo_dict[home_id] = self.new_rating(home_elo, 1, exp_scr)
                self.elo_dict[away_id] = self.new_rating(away_elo, 0, 1 - exp_scr)
            elif result == "L":
                self.elo_dict[home_id] = self.new_rating(home_elo, 0, exp_scr)
                self.elo_dict[away_id] = self.new_rating(away_elo, 1, 1 - exp_scr)
            else:
                continue

            self.elo_date_team = self.elo_date_team.append(
                {
                    "DATE": game_date,
                    "SEASON": season_id,
                    "TEAM_ID": home_id,
                    "ELO": self.elo_dict[home_id],
                },
                ignore_index=True,
            )
            self.elo_date_team = self.elo_date_team.append(
                {
                    "DATE": game_date,
                    "SEASON": season_id,
                    "TEAM_ID": away_id,
                    "ELO": self.elo_dict[away_id],
                },
                ignore_index=True,
            )

        self.edt = self.elo_date_team.copy()
        self.edt = self.edt.sort_values(by="DATE")
        dump = self.edt[
            self.edt.groupby(["SEASON", "TEAM_ID"]).DATE.transform("max")
            == self.edt.DATE
        ]
        dump.to_csv("../prep/data/elos.csv", index=False)


class Salaries:
    def __init__(self):
        self.df = pd.DataFrame()

    def scrape(self, start, end):
        for year in tqdm(range(start, end + 1)):  # 2022
            for page in range(1, 19):  # 18
                try:
                    temp = pd.read_html(
                        f"https://www.espn.com/nba/salaries/_/year/{year}/page/{page}",
                        header=0,
                    )[0]
                    temp["YEAR"] = year
                    self.df = self.df.append(temp, ignore_index=True)
                    sleep(0.6)
                except:
                    continue

    def dump(self):
        self.df = self.df.drop(self.df[self.df["NAME"] == "NAME"].index.values)
        self.df = self.df.drop("RK", axis=1)
        self.df["POS"] = self.df["NAME"].apply(lambda x: x.split(",")[1])
        self.df["NAME"] = self.df["NAME"].apply(lambda x: x.split(",")[0])
        self.df["SALARY"] = self.df["SALARY"].apply(lambda x: x.strip("$"))
        self.df["SALARY"] = self.df["SALARY"].apply(lambda x: x.replace(",", ""))
        self.df["SALARY"] = self.df["SALARY"].astype(np.compat.long)
        self.df.drop(
            self.df[self.df.TEAM == "null Unknown"].index, axis=0, inplace=True
        )
        self.df.TEAM = self.df.TEAM.apply(fix_team_names)
        self.df.reset_index(drop=True, inplace=True)
        self.df.to_csv("../prep/data/salaries.csv", index=False)


class winProbability:
    def __init__(self, team):
        self.team = team
        with open("../prep/models/winprobability/winprobamodel.pkl", "rb") as file:
            self.model = pkl.load(file)

    def prep(self):
        schedule = pd.read_csv("../prep/data/schedule.csv").drop("Time / TV", axis=1)
        self.next_match = (
            schedule[(schedule.Away == self.team) | (schedule.Home == self.team)][
                ["Away", "Home"]
            ]
            .iloc[0:1, :]
            .values
        )
        melo = pd.read_csv("../prep/models/winprobability/data/melo.csv")
        away = (
            melo[melo.TEAM == self.next_match[0][0]].iloc[-1:, :].reset_index(drop=True)
        )
        away.columns = ["AWAY_" + col for col in away.columns]
        home = (
            melo[melo.TEAM == self.next_match[0][1]].iloc[-1:, :].reset_index(drop=True)
        )
        home.columns = ["HOME_" + col for col in home.columns]
        melo = pd.concat([away, home], axis=1)
        melo.drop(["AWAY_W", "AWAY_L", "HOME_W", "HOME_L"], axis=1, inplace=True)
        rankings = pd.read_csv("../prep/data/daily_rankings_cleaned.csv")
        away_rank = (
            rankings[rankings.TEAM == self.next_match[0][0]]
            .iloc[-1:, :]
            .reset_index(drop=True)
        )
        away_rank.drop(["PW", "PL", "PS/G", "PA/G"], axis=1, inplace=True)
        away_rank.columns = ["AWAY_" + col for col in away_rank.columns]
        home_rank = (
            rankings[rankings.TEAM == self.next_match[0][1]]
            .iloc[-1:, :]
            .reset_index(drop=True)
        )
        home_rank.drop(["PW", "PL", "PS/G", "PA/G"], axis=1, inplace=True)
        home_rank.columns = ["HOME_" + col for col in home_rank.columns]
        rankings = pd.concat([away_rank, home_rank], axis=1)

        self.final = pd.concat([melo, rankings], axis=1).drop(
            [
                "HOME_GP",
                "AWAY_GP",
                "AWAY_SEASON",
                "AWAY_MONTH",
                "HOME_SEASON",
                "HOME_TEAM",
                "HOME_MONTH",
                "AWAY_TEAM",
                "HOME_DATE",
                "AWAY_DATE",
                "AWAY_MIN",
                "HOME_MIN",
            ],
            axis=1,
        )

        self.final["HOME_POSS"] = (
            (self.final["HOME_FGA"] - self.final["HOME_OREB"])
            + self.final["HOME_TOV"]
            + (0.44 * self.final["HOME_FTA"])
        )
        self.final["AWAY_POSS"] = (
            (self.final["AWAY_FGA"] - self.final["AWAY_OREB"])
            + self.final["AWAY_TOV"]
            + (0.44 * self.final["AWAY_FTA"])
        )

        self.final["HOME_OFF_RATING"] = 100 * (
            self.final["HOME_PTS"] / self.final["HOME_POSS"]
        )
        self.final["AWAY_OFF_RATING"] = 100 * (
            self.final["AWAY_PTS"] / self.final["AWAY_POSS"]
        )

        self.final["HOME_DEF_RATING"] = 100 * (
            self.final["AWAY_PTS"] / self.final["AWAY_POSS"]
        )
        self.final["AWAY_DEF_RATING"] = 100 * (
            self.final["HOME_PTS"] / self.final["HOME_POSS"]
        )

        self.final["HOME_GP"] = self.final["HOME_W"] + self.final["HOME_L"]
        self.final["AWAY_GP"] = self.final["AWAY_W"] + self.final["AWAY_L"]

        self.final["HOME_WIN_PERC"] = (
            self.final["HOME_W"] / self.final["HOME_GP"] + 0.1
        ) * 100
        self.final["AWAY_WIN_PERC"] = (
            self.final["AWAY_W"] / self.final["AWAY_GP"] + 0.1
        ) * 100

        home_cols = [
            "HOME_PTS",
            "HOME_FGM",
            "HOME_FGA",
            "HOME_3PM",
            "HOME_3PA",
            "HOME_FTM",
            "HOME_FTA",
            "HOME_OREB",
            "HOME_DREB",
            "HOME_REB",
            "HOME_AST",
            "HOME_TOV",
            "HOME_STL",
            "HOME_BLK",
            "HOME_PF",
        ]
        for col in home_cols:
            self.final[col] = self.final[col] / self.final["HOME_GP"]

        away_cols = [
            "AWAY_PTS",
            "AWAY_FGM",
            "AWAY_FGA",
            "AWAY_3PM",
            "AWAY_3PA",
            "AWAY_FTM",
            "AWAY_FTA",
            "AWAY_OREB",
            "AWAY_DREB",
            "AWAY_REB",
            "AWAY_AST",
            "AWAY_TOV",
            "AWAY_STL",
            "AWAY_BLK",
            "AWAY_PF",
        ]

        for col in away_cols:
            self.final[col] = self.final[col] / self.final["AWAY_GP"]

        self.final.drop(["HOME_GP", "AWAY_GP"], axis=1, inplace=True)
        self.final.reset_index(drop=True, inplace=True)

        first = self.final[[col for col in self.final.columns if "HOME" in col]]
        second = self.final[[col for col in self.final.columns if "AWAY" in col]]

        self.final = first.div(second.values)
        self.final.columns = [
            col.split("_")[-1] + "_RATIO" for col in self.final.columns
        ]
        self.final.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.final = self.final.fillna(1)
        self.final.columns = [
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
        ]

    def get_prediction(self):
        self.y_pred = self.model.predict_proba(self.final)
        self.prediction = pd.DataFrame(self.y_pred)
        self.prediction.columns = self.next_match[0]
        return self.prediction


class PERForecast:
    def __init__(self):
        with open("../prep/models/per/perforecastmodel.pkl", "rb") as file:
            self.model = pkl.load(file)

    def get_player_perf_forecast(self):
        # Import and Process
        all_df = pd.read_csv("../prep/data/per.csv")
        all_df.loc[all_df.LAST_NAME == "Yao Ming", "FIRST_NAME"] = "Yao"
        all_df.loc[all_df.LAST_NAME == "Yao Ming", "LAST_NAME"] = "Ming"
        all_df.loc[all_df.LAST_NAME == "Nene", "FIRST_NAME"] = "Nene"
        all_df.loc[all_df.LAST_NAME == "Nene", "LAST_NAME"] = "Nene"
        all_df.loc[all_df.LAST_NAME == "Yi Jianlian", "FIRST_NAME"] = "Yi"
        all_df.loc[all_df.LAST_NAME == "Yi Jianlian", "LAST_NAME"] = "Jianlian"
        all_df["NAME"] = all_df["FIRST_NAME"] + " " + all_df["LAST_NAME"]

        all_df = pd.read_csv("../prep/data/per.csv")
        all_df["NAME"] = all_df["FIRST_NAME"] + " " + all_df["LAST_NAME"]
        all_df.drop(
            [
                "FIRST_NAME",
                "LAST_NAME",
                "P_ID",
                "TEAM_ID",
                "TEAM_ABBREVIATION",
                "GS",
                "MIN",
                "FGM",
                "FGA",
                "FG3M",
                "FG3A",
                "FTM",
                "FTA",
            ],
            axis=1,
            inplace=True,
        )
        all_df
        all_df["REB"] = all_df["REB"] / all_df["GP"]
        all_df["AST"] = all_df["AST"] / all_df["GP"]
        all_df["STL"] = all_df["STL"] / all_df["GP"]
        all_df["BLK"] = all_df["BLK"] / all_df["GP"]
        all_df["TOV"] = all_df["TOV"] / all_df["GP"]
        all_df["PF"] = all_df["PF"] / all_df["GP"]
        all_df["PTS"] = all_df["PTS"] / all_df["GP"]
        all_df["SEASON"] = all_df.SEASON_ID.apply(lambda x: int(x[:4]))

        # Lagging
        labels = all_df[["NAME", "PER", "SEASON"]]
        labels["SEASON"] = labels["SEASON"] - 1
        labels = labels.rename(columns={"PER": "NEXT_PER"})
        all_df = all_df.merge(labels, on=["SEASON", "NAME"], how="left")

        current_season = all_df[(all_df.SEASON == 2021)]
        current_season_check = current_season[["NAME", "PER", "SEASON_ID", "TEAM"]]

        all_df.dropna(inplace=True)
        all_df.reset_index(drop=True, inplace=True)

        check = all_df[["NAME", "PER", "SEASON_ID", "TEAM"]]
        check = check[~check.duplicated()]  # to add later

        all_df.drop(
            ["SEASON_ID", "SEASON", "NAME", "PER", "TEAM"], axis=1, inplace=True
        )
        all_df = all_df[~all_df.duplicated()]

        # Get Predictions
        X = all_df.drop("NEXT_PER", axis=1)
        y = all_df["NEXT_PER"]
        y_pred = self.model.predict(X)
        all_df["y_pred"] = y_pred

        # history
        final = pd.concat([all_df, check], axis=1)
        final = final[["TEAM", "NAME", "SEASON_ID", "PER", "y_pred"]]
        final = final.rename(columns={"y_pred": "PRED"})

        # current
        current_season.drop(
            ["SEASON_ID", "SEASON", "NAME", "PER", "TEAM", "NEXT_PER"],
            axis=1,
            inplace=True,
        )
        current_season["PRED"] = self.model.predict(current_season)
        current_season = pd.concat([current_season, current_season_check], axis=1)
        current_season = current_season[["TEAM", "NAME", "SEASON_ID", "PER", "PRED"]]

        final = pd.concat([final, current_season], axis=0)
        final.dropna(inplace=True)
        final.to_csv("../prep/estimations/perf_forecast.csv", index=False)


class MVPForecast:
    def __init__(self):
        self.advncd_2022 = pd.read_html(
            f"https://www.basketball-reference.com/leagues/NBA_2022_advanced.html",
            header=0,
            match="Advanced",
        )[0]
        self.cands_2022 = pd.read_html(
            f"https://www.basketball-reference.com/friv/mvp.html", header=0
        )[0].drop(["Unnamed: 31", "Prob%"], axis=1)
        with open("../prep/models/mvp/mvpmodel.pkl", "rb") as file:
            self.model = pkl.load(file)

    def prep_and_predict(self):
        self.data_2022 = self.cands_2022.merge(
            self.advncd_2022,
            how="left",
            left_on=["Player", "Team"],
            right_on=["Player", "Tm"],
        ).drop(["W", "L", "Rk_x", "Rk_y", "Tm", "Unnamed: 19", "Unnamed: 24"], axis=1)

        self.df_2022 = self.data_2022.get(["W/L%", "WS", "VORP", "PER", "USG%", "BPM"])
        self.data_2022["Share"] = self.model.predict(self.df_2022)
        self.data_2022 = self.data_2022.sort_values(
            "Predicted_Share", ascending=False
        ).get(
            [
                "Player",
                "Year",
                "Tm",
                "W/L%",
                "WS",
                "VORP",
                "PER",
                "USG%",
                "BPM",
                "Predicted_Share",
            ]
        )
        self.data_2022.to_csv(f"../prep/estimations/mvps/2022_mvp.csv", index=False)
