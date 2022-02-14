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


### ÖRNEK KULLANIM

# kw_list = ['Kobe Bryant', 'Lebron James', 'Stephen Curry']
#
#
# historicaldf = pytrend.get_historical_interest(kw_list, year_start=2019, month_start=10, day_start=1,
#                                                hour_start=0, year_end=2021, month_end=10, day_end=1,
#                                                hour_end=0, geo='US', sleep=1, frequency='daily')
# # gprop='', cat=0
#
# mj_hist = historicaldf['Kobe Bryant']
# sc_hist = historicaldf['Lebron James']
# lj_hist = historicaldf['Stephen Curry']
# sns.lineplot(x=mj_hist.index, y=np.array(mj_hist), color='red')
# sns.lineplot(x=sc_hist.index, y=np.array(sc_hist), color='blue')
# sns.lineplot(x=lj_hist.index, y=np.array(lj_hist), color='green')
# plt.show()


### PROJE İÇİN KULLANIMI

players = pd.read_csv("data/base/merged.csv")
players["NAME"] = players["FIRST_NAME"] + " " + players["LAST_NAME"]
players = players[players.SEASON_ID > "2003-04"]
players.dropna(inplace=True)
players = list(set(players.NAME.to_list()))
players = players + ["Yao Ming", "Nene", "Yi Jianlian"]
players = sorted(players)


base_term = "Lebron James"
# 2021'de en çok aranan kelimelerden biri  olduğu için güzel bir referans noktası oluşturur.



# search_dict = {}
def get_trends(base_term, players):
    # for player in tqdm(players[:1]):
    # random = np.random.random() * 3
    kw_list = [base_term] + [players]

    # 2005 ilk gününden 2022 ilk gününe kadar arama verilerini getir
    historicaldf = pytrend.get_historical_interest(
        kw_list,
        year_start=2005,
        month_start=1,
        day_start=1,
        hour_start=0,
        year_end=2022,
        month_end=1,
        day_end=1,
        hour_end=0,
        geo="US",
        sleep=60,
        frequency="daily",
    )

    historicaldf.to_csv('trends_deneme.csv', index=False)
    # with open("data/base/trends.pkl", "wb") as file:
    #     pkl.dump(search_dict, file)


get_trends(base_term, players)

# with open("data/base/trends.pkl", "rb") as file:
#     trends = pkl.load(file)
#
#
#
# trends = pd.DataFrame(trends)


dataset = []

for i in tqdm(range(len(players))):
    print(players[i])
    random = np.random.random() * 3
    keywords = [players[i]]
    pytrend.build_payload(
        kw_list=keywords,
        cat=0,
        timeframe='2005-01-01 2022-01-01',
        geo='US')
    data = pytrend.interest_over_time()
    if not data.empty:
        data = data.drop(labels=['isPartial'], axis='columns')
        dataset.append(data)
    else:
        print('Empty dataset.')
    sleep(.6 + random)

result = pd.concat(dataset, axis=1)
result.to_csv('data/base/trendtest.csv', index=False)



result['LeBron James'] - result['Dwight Howard']
result['Dwight Howard']
