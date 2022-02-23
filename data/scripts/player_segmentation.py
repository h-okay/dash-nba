import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)


##### Reading Datas
stats = pd.read_csv("data/base/stats.csv")
players = pd.read_csv("data/base/all_players.csv")
salary = pd.read_csv("data/base/salaries.csv")

#### CONVERTING SEASON_ID to YEAR
for i, val in enumerate(stats["SEASON_ID"]):
    stats.loc[i, "YEAR"] = "20" + val[5:]


stats = stats[stats["YEAR"] == "2022"]

### Getting player names from all_players data
stats = stats.merge(players, left_on="PLAYER_ID", right_on="id")
stats[stats.full_name.duplicated()]
# Drop duplicated players keeping first row
stats.drop_duplicates("full_name", keep="first", inplace=True, ignore_index=True)


### Getting player positions from salary data
stats = stats.merge(salary, left_on="full_name", right_on="NAME")
stats.drop_duplicates("full_name", keep="last", inplace=True, ignore_index=True)

stats.head()
stats.drop(
    [
        "PLAYER_ID",
        "SEASON_ID",
        "LEAGUE_ID",
        "TEAM_ID",
        "id",
        "first_name",
        "last_name",
        "is_active",
        "NAME",
        "YEAR_y",
    ],
    axis=1,
    inplace=True,
)
# we delete columns unrelated to segmentation
df = stats.drop(["TEAM_ABBREVIATION", "YEAR_x", "full_name", "TEAM", "POS"], axis=1)


# SCALING
scaler = MinMaxScaler((0, 3))
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df.head(5)

# Finding optimum cluster number
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()
elbow.elbow_value_

## Fınal clustering model
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)
clusters = kmeans.labels_
centers = kmeans.cluster_centers_

segments = pd.DataFrame({"full_name": stats.full_name, "segments": clusters})
segments["segments"] = segments["segments"] + 1
segments.head(20)
segments.shape

### analysis of clusters in position breakdown
stats = stats.merge(segments, on="full_name")

stats.head(20)

stats.groupby("segments").agg(
    {"PLAYER_AGE": "mean", "SALARY": "mean", "PTS": "mean", "full_name": "count"}
)

sgmnt = stats.groupby(["segments", "POS"]).agg(
    {"PLAYER_AGE": "mean", "SALARY": "mean", "PTS": "mean", "full_name": "count"}
)


sns.barplot(x="segments", y="SALARY", hue="POS", data=stats)
plt.show()

sns.scatterplot(x="PTS", y="SALARY", hue="segments", style="segments", data=stats)
plt.show()
