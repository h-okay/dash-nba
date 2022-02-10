import pandas as pd
import requests
import urllib
import os
from bs4 import BeautifulSoup
from time import sleep
from tqdm import tqdm
import glob


per = pd.read_csv("../data/base/per.csv")
per['NAME'] = per['FIRST_NAME'] + " " + per['LAST_NAME']

def get_headshots():
    for team in per.TEAM.unique():
        names = per[(per.TEAM == team) & (per.SEASON_ID == '2021-22')].\
            sort_values(by='PER', ascending=False).reset_index(drop=True)['NAME'].to_list()
        ids = per[(per.TEAM == team) & (per.SEASON_ID == '2021-22')].\
            sort_values(by='PER', ascending=False).reset_index(drop=True)['P_ID'].to_list()
        pers = per[(per.TEAM == team) & (per.SEASON_ID == '2021-22')].\
            sort_values(by='PER', ascending=False).reset_index(drop=True)['PER'].to_list()
        zipped = list(zip(names, ids, pers))

        dirName = f"assets/top/{team}"
        try:
            os.makedirs(dirName)
        except FileExistsError:
            pass

        files = glob.glob(f"dirName/*")
        for f in files:
            os.remove(f)

        for val in tqdm(zipped, desc=team):
            url = f"https://www.nba.com/player/{val[1]}"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            if val[0] == 'Nicolas Claxton':
                urllib.request.urlretrieve(soup.find_all('img',
                                                         alt="Nic Claxton Headshot")[0]['src'],
                                           f"assets/top/{team}/Nicolas Claxton.jpg")
                sleep(.6)
                continue
            if val[0] == 'KJ Martin':
                urllib.request.urlretrieve(soup.find_all('img',
                                                         alt="Kenyon Martin Jr. Headshot")[0]['src'],
                                           f"assets/top/{team}/KJ Martin.jpg")
                sleep(.6)
                continue
            if val[0] == 'Xavier Tillman Sr.':
                urllib.request.urlretrieve(soup.find_all('img',
                                                         alt="Xavier Tillman Headshot")[
                                               0]['src'],
                                           f"assets/top/{team}/Xavier Tillman Sr.jpg")
                sleep(.6)
                continue
            if val[0] == 'Nah\'Shon Hyland':
                urllib.request.urlretrieve(soup.find_all('img',
                                                         alt="Bones Hyland Headshot")[
                                               0]['src'],
                                           f"assets/top/{team}/Nah\'Shon Hyland.jpg")
                sleep(.6)
                continue
            else:
                urllib.request.urlretrieve(soup.find_all('img',
                                                         alt=f"{val[0]} Headshot")[0]['src'],
                                           f"assets/top/{team}/{val[0]}.jpg")
                sleep(.6)

if __name__ == '__main__':
    get_headshots()


