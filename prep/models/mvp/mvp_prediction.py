import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_val_predict,
    GridSearchCV,
)
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from utils.helpers import *
from tqdm import tqdm


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)

year = date.today().year
mvp_cands = pd.read_csv("prep/data/mvp_candidates.csv")

mvp_cands.describe([0.01, 0.05, 0.5, 0.95, 0.99]).T


mvp_cands[mvp_cands["W/L%"].isnull()]


mvp_cands.shape


mvp_cands.groupby("Pos").agg({"MVP": "count"})

mvp_cands.head()
df = mvp_cands.get(
    [
        "3P%",
        "FT%",
        "AST%",
        "STL%",
        "TRB%",
        "BLK%",
        "TOV%",
        "TS%",
        "PER",
        "WS",
        "FTr",
        "BPM",
        "VORP",
        "USG%",
        "W/L%",
        "Share",
    ]
)
df.head()


########  BASE MODEL  ##########
y = df["Share"]
X = df.drop("Share", axis=1)


lgbm = LGBMRegressor()
np.mean(np.sqrt(-cross_val_score(lgbm, X, y, cv=5, scoring="neg_mean_squared_error")))


xgb = XGBRegressor()
np.mean(np.sqrt(-cross_val_score(xgb, X, y, cv=5, scoring="neg_mean_squared_error")))


cat = CatBoostRegressor()
np.mean(np.sqrt(-cross_val_score(cat, X, y, cv=5, scoring="neg_mean_squared_error")))


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame(
        {"Value": model.feature_importances_, "Feature": features.columns}
    )
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(
        x="Value",
        y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False)[0:num],
    )
    plt.title(f"{model} Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


plot_importance(cat.fit(X, y), X)
plot_importance(lgbm.fit(X, y), X)
plot_importance(xgb.fit(X, y), X)

######## HYPER PARAMETRE Optimization ##################

df = mvp_cands.get(["W/L%", "WS", "VORP", "PER", "USG%", "BPM", "Share"])
y = df["Share"]
X = df.drop("Share", axis=1)

cat = CatBoostRegressor()
parameters = {
    "depth": [6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "iterations": [30, 50, 100],
    "early_stopping_rounds": [200],
}
grid = GridSearchCV(estimator=cat, param_grid=parameters, cv=5, n_jobs=-1).fit(X, y)
grid.best_params_


final_cat = cat.set_params(**grid.best_params_).fit(X, y)
np.mean(
    np.sqrt(-cross_val_score(final_cat, X, y, cv=5, scoring="neg_mean_squared_error"))
)  # 0.16

import pickle as pkl

with open("prep/models/mvp/mvpmodel.pkl", "wb") as file:
    pkl.dump(final_cat, file)

###############  2022  Prediction ##################
advncd_2022 = pd.read_html(
    f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html",
    header=0,
    match="Advanced",
)[0]
cands_2022 = pd.read_html(
    f"https://www.basketball-reference.com/friv/mvp.html", header=0
)[0].drop(["Unnamed: 31", "Prob%"], axis=1)

data_2022 = cands_2022.merge(
    advncd_2022, how="left", left_on=["Player", "Team"], right_on=["Player", "Tm"]
).drop(["W", "L", "Rk_x", "Rk_y", "Tm", "Unnamed: 19", "Unnamed: 24"], axis=1)

df_2022 = data_2022.get(["W/L%", "WS", "VORP", "PER", "USG%", "BPM"])

data_2022["Share"] = final_cat.predict(df_2022)

data_2022.sort_values("Share", ascending=False).get(["Player", "Team", "Share"])
data_2022.head()
data_2022.to_csv(f"prep/estimations/mvps/{year}_mvp.csv", index=False)

###############   2018-2021 Prediction  ####################
for i in range(2018, year):
    temp = mvp_cands[mvp_cands["Year"] == i].get(
        ["Player", "Year", "Tm", "W/L%", "WS", "VORP", "PER", "USG%", "BPM", "Share"]
    )

    temp["Predicted_Share"] = final_cat.predict(
        temp.drop(["Share", "Player", "Year", "Tm"], axis=1)
    )

    temp.to_csv(f"prep/estimations/mvps/{i}_mvp.csv", index=False)
