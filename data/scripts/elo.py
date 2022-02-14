import pandas as pd
import pickle as pkl
import warnings
import numpy as np
from tqdm import tqdm

from pandas.core.common import SettingWithCopyWarning

pd.set_option("display.float_format", lambda x: "%.3f" % x)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

all_players = pd.read_csv("../data/base/all_players.csv")
all_teams = pd.read_csv("../data/base/all_teams.csv")
merged = pd.read_csv("../data/base/merged.csv")

with open("data/base/matches.pkl", "rb") as file:
    matches = pkl.load(file)

for i in matches.values():
    cols = i.columns
    break

########### MAÇLARI TEK SATIRA İNDİRGEME
def calculate_elo():
    # Takımın maçı kazanma olasılığını hesapla
    def expected_score(elo_a, elo_b):
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    # Maçın sonucuna göre takımın yeni ELOsunu bul.
    def new_rating(elo_a, score_a, expected_score_a):
        return elo_a + 32 * (score_a - expected_score_a)

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

    ####### ELO HESAPLAMA

    elo_dict = dict(zip(team_ids, elo_ratings))
    elo_date_team = pd.DataFrame(columns=["DATE", "SEASON", "TEAM_ID", "ELO"])

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
    c = edt[edt.groupby(["SEASON", "TEAM_ID"]).DATE.transform("max") == edt.DATE]
    c.to_csv("../data/base/elos.csv", index=False)
