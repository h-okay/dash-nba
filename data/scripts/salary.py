import numpy as np
import pandas as pd
from tqdm import tqdm
from time import sleep

from data.scripts.helpers import fix_team_names


def get_salaries():
    df = pd.DataFrame()
    for year in tqdm(range(2004, 2023)):  # 2022
        for page in range(1, 19):  # 18
            try:
                temp = pd.read_html(
                    f"https://www.espn.com/nba/salaries/_/year/{year}/page/{page}",
                    header=0,
                )[0]
                temp["YEAR"] = year
                df = df.append(temp, ignore_index=True)
                sleep(0.6)
            except:
                continue

    df = df.drop(df[df["NAME"] == "NAME"].index.values)
    df = df.drop("RK", axis=1)
    df["NAME"] = df["NAME"].apply(lambda x: x.split(",")[0])
    df["SALARY"] = df["SALARY"].apply(lambda x: x.strip("$"))
    df["SALARY"] = df["SALARY"].apply(lambda x: x.replace(",", ""))
    df["SALARY"] = df["SALARY"].astype(np.compat.long)
    df.drop(df[df.TEAM == "null Unknown"].index, axis=0, inplace=True)
    df.TEAM = df.TEAM.apply(fix_team_names)
    df.reset_index(drop=True, inplace=True)

    df.to_csv("data/base/salaries.csv", index=False)
    print("Done.")


get_salaries()
