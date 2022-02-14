import pandas as pd
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from time import time
from tqdm import tqdm
import pickle as pkl
from time import sleep

pytrend = TrendReq()

### Players
players = pd.read_csv("data/base/merged.csv")
players["NAME"] = players["FIRST_NAME"] + " " + players["LAST_NAME"]
players = players[players.SEASON_ID > "2003-04"]
players.dropna(inplace=True)
players = list(set(players.NAME.to_list()))
players = players + ["Yao Ming", "Nene", "Yi Jianlian"]
players = sorted(players)

def get_player_trends():
    dataset = []
    for i in tqdm(range(len(players))):
        random = np.random.random() * 3
        keywords = [players[i]]
        pytrend.build_payload(
            kw_list=keywords,
            cat=0,
            timeframe='2004-01-01 2022-01-01',
            geo='US')
        data = pytrend.interest_over_time()
        if not data.empty:
            data = data.drop(labels=['isPartial'], axis='columns')
            dataset.append(data)
        else:
            print('Empty dataset.')
        sleep(.6 + random)

    result = pd.concat(dataset, axis=1).reset_index()
    result.to_csv('data/base/trendtest.csv', index=False)

get_player_trends()
