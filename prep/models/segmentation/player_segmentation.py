import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import AgglomerativeClustering

import plotly.express as px
import plotly.io as pio

# pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)
# pd.set_option("display.width", 500)


##### Reading Datas
salary = pd.read_csv("prep/data/salaries.csv")
per_ = pd.read_csv("prep/data/per.csv")

#### CONVERTING SEASON_ID to YEAR
for i, val in enumerate(per_["SEASON_ID"]):
    per_.loc[i, "YEAR"] = "20" + val[5:]

####
# Concating FIRST and LAST NAME
per_["NAME"] = per_["FIRST_NAME"] + " " + per_["LAST_NAME"]
per_.insert(0, "NAME", per_.pop("NAME"))  # getting player names to first column


per = per_.copy()
per = per[per["YEAR"] == "2022"]
per.shape

### Getting player POSITIONS and SALARY from salary data
per = (
    per.merge(salary, on="NAME", how="inner")
    .drop_duplicates("P_ID", keep="last", ignore_index=True)
    .reset_index(drop=True)
)

### DROP duplicate and unnecessary columns
per.drop(
    [
        "FIRST_NAME",
        "LAST_NAME",
        "TEAM_ABBREVIATION",
        "P_ID",
        "SEASON_ID",
        "TEAM_ID",
        "TEAM_x",
        "YEAR_x",
    ],
    axis=1,
    inplace=True,
)

per.rename(columns={"TEAM_y": "TEAM", "YEAR_y": "YEAR"}, inplace=True)

# we delete unrelated columns to segmentation
df = per.drop(
    [col for col in per.columns if col not in ["AGE", "PTS", "MPG", "PER"]], axis=1
)


####### SCALING
scaler = RobustScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df.head()


########### HIERARCHICAL CLUSTERING ####################
# Finding optimum cluster number with HIEARCHICAL CLUSTERING
hc_complete = linkage(df, "complete")
hc_average = linkage(df, "average")

plt.figure(figsize=(10, 5))
plt.title("Hierarchical Clustering Dendogram")
plt.xlabel("Observation Unit")
plt.ylabel("Distances")
dendrogram(hc_complete, leaf_font_size=15)
plt.show()

### FINAL MODEL HIERARCHICAL CLUSTERING
cluster = AgglomerativeClustering(n_clusters=5)
clusters = cluster.fit_predict(df)
segments = pd.DataFrame({"NAME": per.NAME, "H_cluster_no": clusters})
segments["H_cluster_no"] = segments["H_cluster_no"] + 1
per = per.merge(segments, on="NAME")


### analysis of Segments
sgmnts = {
    "1": "Average",
    "4": "Worst",
    "2": "Overperforming",
    "3": "Underperforming",
    "5": "Best",
}
per["Segment"] = per["H_cluster_no"].astype(str).map(sgmnts)

per.groupby("Segment").agg(
    {
        "AGE": "mean",
        "PTS": "mean",
        "MPG": "mean",
        "PER": "mean",
        "SALARY": "mean",
        "NAME": "count",
    }
)

# per.drop('H_cluster_no', axis=1, inplace=True)
# per.to_csv('prep/estimations/segmentation.csv', index=False)
# ##### Intractive PLOT
# ############ PER - SALARY SCATTER ################
# pio.renderers.default = "browser"
# fig = px.scatter(per, x="PER", y="SALARY", color='cluster_no', hover_name="NAME", trendline="ols")
# fig.update_traces(marker_size=10)
# fig.show()
#
# ############ PTS - SALARY SCATTER ##################
# pio.renderers.default = "browser"
# fig = px.scatter(per, x="PTS", y="SALARY", color='cluster_no', hover_name="NAME", trendline="ols")
# fig.update_traces(marker_size=10)
# fig.show()
#

#############  3D SCATTER  ####################
team = "Miami Heat"
pio.renderers.default = "browser"
per["Segment"] = per["Segment"].astype(str)
fig = px.scatter_3d(
    per,
    x="PER",
    y="MPG",
    z="AGE",
    hover_data=["TEAM", "POS"],
    hover_name="NAME",
    color="Segment",
    symbol=np.where(per["TEAM"] == team, team, "Other"),
    symbol_map={team: "diamond", "Other": "cross"},
    size_max=10,
)
fig.show()
