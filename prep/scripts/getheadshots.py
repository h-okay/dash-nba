import pandas as pd
import requests
import urllib
import os
from bs4 import BeautifulSoup
from time import sleep
from tqdm import tqdm
import glob
from PIL import Image
from prep.scripts.classes import print_done





def get_headshots():
    per = pd.read_csv("prep/data/per.csv")
    per["NAME"] = per["FIRST_NAME"] + " " + per["LAST_NAME"]
    pbar = tqdm(sorted(per.TEAM.unique()), position=0, leave=False)
    for team in pbar:
        pbar.set_description(f"{team}")
        names = (
            per[(per.TEAM == team) & (per.SEASON_ID == "2021-22")]
            .sort_values(by="PER", ascending=False)
            .reset_index(drop=True)["NAME"]
            .to_list()
        )
        ids = (
            per[(per.TEAM == team) & (per.SEASON_ID == "2021-22")]
            .sort_values(by="PER", ascending=False)
            .reset_index(drop=True)["P_ID"]
            .to_list()
        )
        pers = (
            per[(per.TEAM == team) & (per.SEASON_ID == "2021-22")]
            .sort_values(by="PER", ascending=False)
            .reset_index(drop=True)["PER"]
            .to_list()
        )
        zipped = list(zip(names, ids, pers))

        dirName = f"assets/top/{team}"
        try:
            os.makedirs(dirName)
        except FileExistsError:
            pass

        files = glob.glob(f"dirName/*")
        for f in files:
            os.remove(f)

        for val in zipped:
            print(val[0])
            url = f"https://www.nba.com/player/{val[1]}"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            if val[0] == "Nicolas Claxton":
                urllib.request.urlretrieve(
                    soup.find_all("img", alt="Nic Claxton Headshot")[0]["src"],
                    f"assets/{team}/Nicolas Claxton.jpg",
                )
                sleep(0.6)
                continue
            if val[0] == "KJ Martin":
                urllib.request.urlretrieve(
                    soup.find_all("img", alt="Kenyon Martin Jr. Headshot")[0]["src"],
                    f"assets/{team}/KJ Martin.jpg",
                )
                sleep(0.6)
                continue
            if val[0] == "Xavier Tillman Sr.":
                urllib.request.urlretrieve(
                    soup.find_all("img", alt="Xavier Tillman Headshot")[0]["src"],
                    f"assets/{team}/Xavier Tillman Sr.jpg",
                )
                sleep(0.6)
                continue
            if val[0] == "Nah'Shon Hyland":
                urllib.request.urlretrieve(
                    soup.find_all("img", alt="Bones Hyland Headshot")[0]["src"],
                    f"assets/{team}/Nah'Shon Hyland.jpg",
                )
                sleep(0.6)
                continue
            else:
                urllib.request.urlretrieve(
                    soup.find_all("img", alt=f"{val[0]} Headshot")[0]["src"],
                    f"assets/{team}/{val[0]}.jpg",
                )
                sleep(0.6)

def convert_to_png():
    all_teams = pd.read_csv("prep/data/all_teams.csv")
    team_list = all_teams.full_name.unique()
    pbar = tqdm(team_list, position=0, leave=False)
    for team in pbar:
        pbar.set_description(f"{team}")
        links = glob.glob(f"dashboard/assets/{team}/*")
        for link in links:
            with Image.open(link) as im1:
                im1.save(link[:-3]+"png")

    for team in pbar:
        pbar.set_description(f"{team}")
        links = glob.glob(f"dashboard/assets/{team}/*.jpg")
        for link in links:
            os.remove(link)



if __name__ == "__main__":
    print_done('Getting headshots')
    get_headshots()
    convert_to_png()
    print("[DONE]")
