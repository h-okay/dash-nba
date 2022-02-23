import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px
import plotly.io as pio

pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)


##### Reading Datas
salary = pd.read_csv("data/base/salaries.csv")
per_ = pd.read_csv("data/base/per.csv")

#### CONVERTING SEASON_ID to YEAR
for i, val in enumerate(per_["SEASON_ID"]):
    per_.loc[i, "YEAR"] = "20"+val[5:]

####
# Concating FIRST and LAST NAME's

per_["NAME"] = per_["FIRST_NAME"] + " " + per_["LAST_NAME"]
per_.insert(0, "NAME", per_.pop('NAME')) # getting player names to first column

per = per_.copy()
per = per[per["YEAR"] == "2022"]
per.shape

### Getting player POSITIONS and SALARY from salary data
per = per.merge(salary, on="NAME").drop_duplicates("P_ID", keep="last", ignore_index=True).reset_index(drop=True)
per.shape
per.head(10)



per.drop(["FIRST_NAME", "LAST_NAME", "P_ID", "SEASON_ID", "TEAM_ID",
         "first_name", "last_name", "is_active", "NAME", "YEAR_y"], axis=1, inplace=True)
# we delete columns unrelated to segmentation
df = stats.drop(["TEAM_ABBREVIATION", "YEAR_x", "full_name",
          "TEAM", "POS"], axis=1)


# SCALING
scaler = MinMaxScaler((0,3))
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df.head(10)

# Finding optimum cluster number with KMEANS
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()
elbow.elbow_value_

## Fınal clustering model
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)
clusters = kmeans.labels_
centers = kmeans.cluster_centers_

segments = pd.DataFrame({"full_name": stats.full_name, "cluster_no": clusters})
segments["cluster_no"] = segments["cluster_no"] + 1
segments.head(20)
segments.shape

# Finding optimum cluster number with HIEARCHICAL CLUSTERING
hc_complete = linkage(df, "complete")
hc_average = linkage(df, "average")

plt.figure(figsize=(10, 5))
plt.title("Hierarchical Clustering Dendogram")
plt.xlabel("Observation Unit")
plt.ylabel("Distances")
dendrogram(hc_complete,
           leaf_font_size=15)
plt.show()

### FINAL MODEL HIERARCHICAL CLUSTERING
cluster = AgglomerativeClustering(n_clusters=4)
clusters = cluster.fit_predict(df)
segments = pd.DataFrame({"full_name": stats.full_name, "cluster_no": clusters})
segments["cluster_no"] = segments["cluster_no"] + 1


### analysis of clusters in position breakdown
stats = stats.merge(segments, on="full_name")

stats.head(20)

stats.groupby("cluster_no").agg({"PLAYER_AGE": "mean",
                               "SALARY" : "mean",
                               "PTS" : "mean",
                                "full_name" : "count"})


sgmnt = {"1" : "High",
         "2" : "Low",
         "3" : "Mid"}

stats["segments"] = stats["cluster_no"].map(sgmnt)



##### Intractive PLOT
pio.renderers.default = "browser"
stats["cluster_no"] = stats["cluster_no"].astype(str)
fig = px.scatter(stats, x="PTS", y="SALARY", color='cluster_no', hover_name="full_name")
fig.update_traces(marker_size=10)
fig.show()

pio.renderers.default = "browser"
stats["cluster_no"] = stats["cluster_no"].astype(str)
fig = px.histogram(stats, x="POS", y="SALARY", color='cluster_no', hover_name="full_name")
fig.show()

stats.head()
