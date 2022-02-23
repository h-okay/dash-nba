## !pip install imblearn

import pandas as pd
import numpy as np
import seaborn as sns
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score
from data.scripts.helpers import *
from tqdm import tqdm


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)



###############
# DATA PREPARATION
###############
mvp = pd.read_csv("data/base/mvps.csv")

player_stats = pd.read_csv("data/base/per.csv")


####
# SEASON_ID 'nin Yıl olarak kayıt edilmesi
####
for i, val in enumerate(player_stats["SEASON_ID"]):
    player_stats.loc[i, "YEAR"] = "20"+val[5:]


####
# Isim ve Soyisimlerin birlesip PLAYER olarak kayıt edilmesi
####
player_stats["PLAYER"] = player_stats["FIRST_NAME"] + " " + player_stats["LAST_NAME"]
player_stats.insert(0, "PLAYER", player_stats.pop('PLAYER')) # Oyuncu isimlerinin ilk sütuna alınması.

player_stats.shape
mvp.shape
player_stats.head()


#####
# Yılların eşit olduğu oyuncuların MVP değişkenine 1 yazılması
df = player_stats.merge(mvp, on="PLAYER", how="left")
df["YEAR_y"]=df.YEAR_y.fillna(0).astype(int).astype(str)
df["MVP"]= np.where(df["YEAR_x"] == df["YEAR_y"], 1,0)
df.head(100)

####
# Gereksiz Sütunların silinmesi
df.drop(["FIRST_NAME", "LAST_NAME", "P_ID", "TEAM_ID", "SEASON_ID",  "TEAM_ABBREVIATION", "YEAR_y",
                      "POS", "TEAM_y", "TEAM_x", "FG%_y"], axis=1, inplace=True)
df.rename(columns={"FG%_x":"FG%",
                    "YEAR_x" : "YEAR"}, inplace=True)
df.head()



############################
# EXPLORATORY DATA ANALYSIS
############################
df.info()

df.isnull().sum()
df[df["PLAYER"].isnull()]
# Isimdeki NaN değerli oyuncu MVP seçilmediği için veri setinden siliyorum.
df.dropna(subset=["PLAYER"], inplace=True)


# MVP seçilenlerin sayısı, ortalama ve standart sapma kontrolu
df.groupby(["MVP"]).agg({"PER": "describe",
                                      "AGE": "describe",
                                      "MIN": "describe"})


# MVP seçilen oyuncu sayısı grafiği
df.groupby("MVP").size().plot(kind="pie")
plt.show()


######
# FEATURE ENGINEERING

df["PPG"] = df["PTS"] / df["GP"]
df["RPG"] = df["REB"] / df["GP"]
df["APG"] = df["AST"] / df["GP"]
df["BLKPG"] = df["BLK"] / df["GP"]


######
# SCALING  - MANUAL STANDART SCALER
val = df[df["YEAR"] > "2021"]
val["YEAR"].head()
df = df[df["YEAR"] != "2022"]
mvps = df["MVP"].to_list()
df = df.groupby("YEAR").apply(lambda x: (x - x.mean()) / x.std())
df["MVP"] = mvps
df.head()


######
# BASE MODEL
df.drop("PLAYER", axis=1, inplace=True)
df.isnull().sum()
df["vop"].describe([0.01, 0.05, 0.5, 0.7, 0.95, 0.99]).T
df[df["vop"].isna() & df["MVP"] == 1]["vop"] = df["vop"].median()
df.isnull().sum()
df.dropna(inplace=True)
X = df.drop("MVP", axis=1)
y = df["MVP"]

# SMOTE
sm = SMOTE(sampling_strategy=0.5)
X_os, y_os = sm.fit_resample(X, y)

X_os.shape
sum(y_os == 1)
sum(y_os == 0)

y_os[y_os == 1].shape[0] / y_os.shape[0]

y_os.head()
y_os.tail()
# Model
lgbm = LGBMClassifier()
xgb = XGBClassifier()
rf = RandomForestClassifier()
loj_model = LogisticRegression()

df.head()

cv_results = cross_validate(lgbm, X_os, y_os, cv=5, scoring=["roc_auc", "f1", "precision", "recall"])
print(f"AUC: {round(cv_results['test_roc_auc'].mean(),4)} (LGBM) ")
print(f"F1_Score: {round(cv_results['test_f1'].mean(), 4)} (LGBM) ")
print(f"Precision: {round(cv_results['test_precision'].mean(), 4)} (LGBM) ")
print(f"Recall: {round(cv_results['test_recall'].mean(), 4)} (LGBM) ")

cv_results = cross_validate(xgb, X_os, y_os, cv=5, scoring=["roc_auc", "f1", "precision", "recall"])
print(f"AUC: {round(cv_results['test_roc_auc'].mean(),4)} (XGBOOST) ")
print(f"F1_Score: {round(cv_results['test_f1'].mean(), 4)} (XGBOOST) ")
print(f"Precision: {round(cv_results['test_precision'].mean(), 4)} (XGBOOST) ")
print(f"Recall: {round(cv_results['test_recall'].mean(), 4)} (XGBOOST) ")

cv_results = cross_validate(rf, X_os, y_os, cv=5, scoring=["roc_auc", "f1", "precision", "recall"])
print(f"AUC: {round(cv_results['test_roc_auc'].mean(),4)} (RF) ")
print(f"F1_Score: {round(cv_results['test_f1'].mean(), 4)} (RF) ")
print(f"Precision: {round(cv_results['test_precision'].mean(), 4)} (RF) ")
print(f"Recall: {round(cv_results['test_recall'].mean(), 4)} (RF) ")

cv_results = cross_validate(loj_model, X_os, y_os, cv=5, scoring=["roc_auc", "f1", "precision", "recall"])
print(f"AUC: {round(cv_results['test_roc_auc'].mean(),4)} (LOJ) ")
print(f"F1_Score: {round(cv_results['test_f1'].mean(), 4)} (LOJ) ")
print(f"Precision: {round(cv_results['test_precision'].mean(), 4)} (LOJ) ")
print(f"Recall: {round(cv_results['test_recall'].mean(), 4)} (LOJ) ")

val.head()









