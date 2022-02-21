import pandas as pd
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from time import time
from tqdm import tqdm
import pickle as pkl
from time import sleep
import datetime

pytrend = TrendReq()

### Players
players = pd.read_csv("data/base/merged.csv")
players["NAME"] = players["FIRST_NAME"] + " " + players["LAST_NAME"]
players = players[players.SEASON_ID > "2003-04"]
players.dropna(inplace=True)
players = list(set(players.NAME.to_list()))
players = players + ["Yao Ming", "Nene", "Yi Jianlian"]
players = sorted(players)

# def get_player_trends():
#     dataset = []
#     for i in tqdm(range(len(players))):
#         random = np.random.random() * 3
#         keywords = [players[i]]
#         pytrend.build_payload(
#             kw_list=keywords,
#             cat=0,
#             timeframe='2005-01-01 2022-01-01',
#             geo='US')
#         data = pytrend.interest_over_time()
#         if not data.empty:
#             data = data.drop(labels=['isPartial'], axis='columns')
#             dataset.append(data)
#         else:
#             print('Empty dataset.')
#         sleep(.6 + random)
#
#     result = pd.concat(dataset, axis=1).reset_index()
#     result.to_csv('data/base/trendtest.csv', index=False)
#
# get_player_trends()
#
#
# a = pd.read_csv('data/base/trendtest.csv')
# datelist = pd.date_range(start='2005-01-01', end='2022-01-01', freq='MS').tolist()
# a.index = datelist


def get_trends(base_term, kws):
    search_dict = {}
    for player in tqdm(players):
        print()
        random = np.random.random() * 3
        kw_list = [base_term, player]

        historicaldf = pytrend.get_historical_interest(
            kw_list,
            year_start=2004,
            month_start=1,
            year_end=2022,
            month_end=1,
            geo="US",
            sleep=1 + random,
            frequency="daily",
        )
        if not historicaldf.empty:
            search_dict[player] = historicaldf[player]
        else:
            print("Empty dataset.")
        sleep(0.6 + random)
    return search_dict


get_trends("Lebron James", players)

with open("data/base/trends.pkl", "wb") as file:
    pkl.dump(search_dict, file)
