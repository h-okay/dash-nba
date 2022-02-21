import pandas as pd
from data.scripts.helpers import fix_team_names


def get_mvps():
    mvp = pd.read_html("http://www.espn.com/nba/history/awards/_/id/33", header=1)[0]
    mvp = mvp.iloc[:, :-1]

    mvp.drop(mvp[mvp.values == "No stats available."].index, inplace=True)
    mvp = mvp[mvp["YEAR"] > 2003]
    mvp.TEAM = mvp.TEAM.apply(fix_team_names)
    mvp.reset_index(drop=True, inplace=True)

    mvp.to_csv("data/base/mvps.csv", index=False)
    print("Done.")


get_mvps()
