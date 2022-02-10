import pandas as pd
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from time import time
from tqdm import tqdm

pytrend = TrendReq()


### ÖRNEK KULLANIM

kw_list = ['Kobe Bryant', 'Lebron James', 'Stephen Curry']


historicaldf = pytrend.get_historical_interest(kw_list, year_start=2019, month_start=10, day_start=1,
                                               hour_start=0, year_end=2021, month_end=10, day_end=1,
                                               hour_end=0, geo='US', sleep=1, frequency='daily')
# gprop='', cat=0

mj_hist = historicaldf['Kobe Bryant']
sc_hist = historicaldf['Lebron James']
lj_hist = historicaldf['Stephen Curry']
sns.lineplot(x=mj_hist.index, y=np.array(mj_hist), color='red')
sns.lineplot(x=sc_hist.index, y=np.array(sc_hist), color='blue')
sns.lineplot(x=lj_hist.index, y=np.array(lj_hist), color='green')
plt.show()


### PROJE İÇİN KULLANIMI

players = pd.read_csv('data/base/merged.csv')
players['NAME'] = players['FIRST_NAME'] + " " + players['LAST_NAME']
players = players[players.SEASON_ID > '2003-04']
players = list(set(players.NAME.to_list()))
players = players + ['Yao Ming', 'Nene', 'Yi Jianlian']



base_term = 'Lebron James'  # 2021'de en çok aranan kelimelerden biri  olduğu
# için güzel bir referans noktası oluşturur.
search_dict = {}

start = time()

for player in tqdm(player_names[:10]):
    kw_list = [base_term, player]

    # 2005 ilk gününden 2022 ilk gününe kadar arama verilerini getir
    historicaldf = pytrend.get_historical_interest(kw_list, year_start=2005, month_start=1, day_start=1,
                                                   hour_start=0, year_end=2022, month_end=1, day_end=1,
                                                   hour_end=0, geo='US', sleep=1, frequency='daily')

    search_dict[player] = historicaldf[player]

end = time()

print('Time Passed (for 10 players):', end-start)  # 10 oyuncu için 468 saniye sürdü.
