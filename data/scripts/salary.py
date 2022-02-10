import numpy as np
import pandas as pd
from tqdm import tqdm
from time import sleep

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)

df_ = pd.DataFrame()
temp = pd.DataFrame()
for year in tqdm(range(2004,2022)): #2022
    for page in range(1,19): #18
        try:
            temp = pd.read_html(f"http://www.espn.com/nba/salaries/_/year/{year}/page/{page}", header=0)[0]
            temp["YEAR"] = year
            df_ = df_.append(temp,ignore_index=True)
            sleep(.6)
        except:
            continue

df = df_.copy()
df


df = df.drop(df[df["NAME"] == "NAME"].index.values)
df = df.drop("RK", axis=1)
df["NAME"] = df["NAME"].apply(lambda x: x.split(",")[0])
df["SALARY"] = df["SALARY"].apply(lambda x : x.strip("$"))
df["SALARY"] = df["SALARY"].apply(lambda x : x.replace(",", ""))

df["SALARY"] = df["SALARY"].astype(np.compat.long)


df.drop(df[df.TEAM == "null Unknown"].index, axis=0, inplace=True)

df.reset_index(drop=True, inplace=True)

df.to_csv("data/base/salaries.csv", index=False)


