# # PER DATASININ OKUNMASI
#
# per_data = pd.read_csv("data/base/per.csv")
# teams = list(per_data["TEAM_ABBREVIATION"].unique())
#
# # ELO DATASININ OKUNMASI VE IDLERE GÖRE TAKIM İSMİ KISALTMALARININ OLUŞTURULMASI
#
# mapper_prep = per_data.groupby('TEAM_ABBREVIATION')['TEAM_ID'].\
#     unique().apply(lambda x: x[0]).reset_index()
#
# mapper = dict(zip(mapper_prep['TEAM_ID'], mapper_prep['TEAM_ABBREVIATION']))
#
# elos = pd.read_csv("data/base/elos.csv")
#
# elos['TEAM_ABBREVIATION'] = elos['TEAM_ID'].apply(lambda x: mapper[x])
#
#
# ## SEZONLARA GÖRE PER DEĞİŞKENİNİN ÜRETİLMESİ
#
# per_data.groupby('SEASON_ID')["TEAM_ABBREVIATION"].nunique()
# # 2003-2004 senesinde 29 takım var. Bu seneden sonra her sene 30 takım bulunmakta. Dolayısıyla
# # eğitim için 2004-05 sezonu sonrası kullanılacaktır.
#
# eda.pandas_view_set()
#
# start_year = 2006
# end_year = 2022
# prev_year_per_avg = {}
# prev_pred_year = {}
#
#
#
# for year in range(start_year, end_year+1):
#
#     pred_year = year
#     train_year = pred_year-1
#     season = str(train_year-1) + "-" + str(train_year)[-2:]
#     test_season = str(year-1) + "-" + str(year)[-2:]
#
#
#     for team in teams:
#         target_rows = per_data[(per_data['SEASON_ID'] == season)
#                                            & (per_data['TEAM_ABBREVIATION'] == team)]
#         prev_year_per_avg[f"{season} {team}"] = np.round(target_rows['PER'].mean(), 2)
#         # Tahmin edilen sezon değişkeninin üretilmesi
#         prev_pred_year[f"{season} {team}"] = test_season
#
# ## SEZONLARA GÖRE ELO DEĞİŞKENİNİN ÜRETİLMESİ
#
# prev_year_elos = {}
#
# for team in teams:
#     for season in elos['SEASON'].unique():
#         if season > '2003-04':
#             prev_year_elos[f"{season} {team}"] = np.round(elos[(elos['TEAM_ABBREVIATION'] == team)
#             & (elos['SEASON']==season)]['ELO'].values[0], 2)

# PLAYOFFS DATASININ OKUNMASI

# import pandas as pd
# from vboUtil import eda

playoffs = pd.read_csv("data/est/mlready.csv")
playoffs = playoffs[
    playoffs["SEASON"] != "2003-04"
]  # 2003-04'te 29 takım var. Uğraşmaya gerek yok.

east_conf = [
    "Miami Heat",
    "Chicago Bulls",
    "Philadelphia 76ers",
    "Cleveland Cavaliers",
    "Milwaukee Bucks",
    "Boston Celtics",
    "Toronto Raptors",
    "Brooklyn Nets",
    "Charlotte Hornets",
    "Atlanta Hawks",
    "Washington Wizards",
    "New York Knicks",
    "Indiana Pacers",
    "Detroit Pistons",
    "Orlando Magic",
]

playoffs["EAST"] = playoffs["TEAM"].isin(east_conf)


eda.pandas_view_set()
playoffs.head()

## Yeni Değişkenler Üretilmesi

playoffs["FGP"] = playoffs["FGM"] / playoffs["FGA"]
playoffs["FTP"] = playoffs["FTM"] / playoffs["FTA"]
playoffs["FG3P"] = playoffs["FG3M"] / playoffs["FG3A"]

playoffs["ORP"] = playoffs["OREB"] / playoffs["REB"]
playoffs["DRP"] = playoffs["DREB"] / playoffs["REB"]

## STATLARIN YILLARA GÖRE TRENDİNİN İNCELENMESİ


def minimize():
    import seaborn as sns
    import matplotlib.pyplot as plt

    def trend_analyser(data, stat):

        grp = po_train.groupby("SEASON")[stat].mean()
        sns.lineplot(x=grp.index, y=grp.values)
        plt.xticks(rotation=90)
        plt.rcParams["figure.figsize"] = (10, 10)
        plt.show()

    trend_analyser(playoffs, "PER")

    # Yıllar ilerledikçe DRP oranı artıyor, ORP oranı azalıyor. Modelde kullanılması sakıncalı.


# STAGE'E GÖRE DATA HAZIRLAYAN FONKSİYON

drop_list = ["OREB", "DREB", "ORP", "DRP"]


def champ_data_prepare(data, stage, drop_list=None):

    y_data = data[stage]
    X_data = data.drop(
        [
            "TEAM",
            "SEASON",
            "Quarter",
            "Playoff",
            "Semi",
            "Final",
            "Champion",
            "SCORE",
            "CONF",
            "DIV",
            "HOME",
            "ROAD",
            "OT",
            "STREAK",
            "LAST 10",
            "TEAM_ID",
            "EAST",
        ],
        axis=1,
    )

    if drop_list != None:
        X_data.drop(drop_list, axis=1, inplace=True)

    X_y = pd.concat([X_data, y_data], axis=1)
    return X_y


## Train-Test Ayrılması

po_train = playoffs[playoffs["SEASON"] < "2020-21"]
po_test = playoffs[playoffs["SEASON"] == "2020-21"]


# H2O Serverının initialize edilmesi

import h2o
from h2o.automl import H2OAutoML

h2o.init()

# Metriklerin import edilmesi

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

##### PLAYOFFS #####

stage = "Playoff"

train = champ_data_prepare(po_train, stage, drop_list)
test = champ_data_prepare(po_test, stage, drop_list)


df = h2o.H2OFrame(train)
df_test = h2o.H2OFrame(test)
factorslist = [stage]
df[factorslist] = df[factorslist].asfactor()

y = stage
X = df.columns
X.remove(y)

aml = H2OAutoML(max_models=10, seed=1)
aml.train(x=X, y=y, training_frame=df)

# lb = aml.leaderboard
# lb.head(rows=lb.nrows)

# leader_model = aml.leader
# leader_model.model_performance(df_test)

# leader_model.varimp_plot()

y_pred = aml.leader.predict(df_test)
y_pred = y_pred.as_data_frame()["p1"]

pred_concat = pd.concat(
    [po_test[["EAST", "TEAM", "Playoff"]].reset_index(), y_pred], axis=1
)

po_pred = (
    pred_concat.sort_values(["EAST", "p1"], ascending=False).groupby("EAST").head(8)
)

po_test[f"{stage}_pred"] = po_test["TEAM"].isin(po_pred["TEAM"])

##### PLAYOFFS #####


##### QUARTERS #####

stage = "Quarter"
pre_stage = "Playoff"

po_train_q = po_train[po_train[pre_stage] == 1]
train = champ_data_prepare(po_train_q, stage, drop_list)
po_test_q = po_test[po_test[f"{pre_stage}_pred"] == 1]
team_names = po_test_q["TEAM"]
test = champ_data_prepare(po_test_q, stage, drop_list)

df = h2o.H2OFrame(train)
df_test = h2o.H2OFrame(test)
factorslist = [stage]
df[factorslist] = df[factorslist].asfactor()

y = stage
X = df.columns
X.remove(y)

aml = H2OAutoML(max_models=10, seed=1)
aml.train(x=X, y=y, training_frame=df)

# lb = aml.leaderboard
# lb.head(rows=lb.nrows)

# leader_model = aml.leader
# leader_model.model_performance(df_test)

# leader_model.varimp_plot()

y_pred = aml.leader.predict(df_test)
y_pred = y_pred.as_data_frame()["p1"]

pred_concat = pd.concat([team_names.reset_index(), y_pred], axis=1)
winners = [
    "Philadelphia 76ers",
    "Utah Jazz",
    "Milwaukee Bucks",
    "Atlanta Hawks",
    "Denver Nuggets",
    "Los Angeles Clippers",
    "Phoenix Suns",
    "Brooklyn Nets",
]
po_test[f"{stage}_pred"] = po_test["TEAM"].isin(winners)

##### QUARTERS #####


##### SEMI #####

stage = "Semi"
pre_stage = "Quarter"

po_train_q = po_train[po_train[pre_stage] == 1]
train = champ_data_prepare(po_train_q, stage, drop_list)
po_test_q = po_test[po_test[f"{pre_stage}_pred"] == 1]
team_names = po_test_q["TEAM"]
test = champ_data_prepare(po_test_q, stage, drop_list)


df = h2o.H2OFrame(train)
df_test = h2o.H2OFrame(test)
factorslist = [stage]
df[factorslist] = df[factorslist].asfactor()

y = stage
X = df.columns
X.remove(y)

aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=X, y=y, training_frame=df)

# lb = aml.leaderboard
# lb.head(rows=lb.nrows)

# leader_model = aml.leader
# leader_model.model_performance(df_test)

# leader_model.varimp_plot()

y_pred = aml.leader.predict(df_test)
y_pred = y_pred.as_data_frame()["p1"]

pred_concat = pd.concat([team_names.reset_index(), y_pred], axis=1)
winners = [
    "Philadelphia 76ers",
    "Milwaukee Bucks",
    "Los Angeles Clippers",
    "Phoenix Suns",
]
po_test[f"{stage}_pred"] = po_test["TEAM"].isin(winners)

##### SEMI #####


##### FINAL #####

stage = "Final"
pre_stage = "Semi"

po_train_q = po_train[po_train[pre_stage] == 1]
train = champ_data_prepare(po_train_q, stage, drop_list)
po_test_q = po_test[po_test[f"{pre_stage}_pred"] == 1]
team_names = po_test_q["TEAM"]
test = champ_data_prepare(po_test_q, stage, drop_list)


df = h2o.H2OFrame(train)
df_test = h2o.H2OFrame(test)
factorslist = [stage]
df[factorslist] = df[factorslist].asfactor()

y = stage
X = df.columns
X.remove(y)

aml = H2OAutoML(max_models=10, seed=1)
aml.train(x=X, y=y, training_frame=df)

# lb = aml.leaderboard
# lb.head(rows=lb.nrows)

# leader_model = aml.leader
# leader_model.model_performance(df_test)

# leader_model.varimp_plot()

y_pred = aml.leader.predict(df_test)
y_pred = y_pred.as_data_frame()["p1"]

pred_concat = pd.concat([team_names.reset_index(), y_pred], axis=1)
winners = ["Philadelphia 76ers", "Phoenix Suns"]
po_test[f"{stage}_pred"] = po_test["TEAM"].isin(winners)

##### FINAL #####


##### CHAMPION #####

stage = "Champion"
pre_stage = "Final"

po_train_q = po_train[po_train[pre_stage] == 1]
train = champ_data_prepare(po_train_q, stage, drop_list)
po_test_q = po_test[po_test[f"{pre_stage}_pred"] == 1]
team_names = po_test_q["TEAM"]
test = champ_data_prepare(po_test_q, stage, drop_list)


df = h2o.H2OFrame(train)
df_test = h2o.H2OFrame(test)
factorslist = [stage]
df[factorslist] = df[factorslist].asfactor()

y = stage
X = df.columns
X.remove(y)

aml = H2OAutoML(max_models=10, seed=1)
aml.train(x=X, y=y, training_frame=df)

# lb = aml.leaderboard
# lb.head(rows=lb.nrows)

# leader_model = aml.leader
# leader_model.model_performance(df_test)

# leader_model.varimp_plot()

y_pred = aml.leader.predict(df_test)
y_pred = y_pred.as_data_frame()["p1"]

pred_concat = pd.concat([team_names.reset_index(), y_pred], axis=1)
winners = ["Philadelphia 76ers", "Phoenix Suns"]
po_test[f"{stage}_pred"] = po_test["TEAM"].isin(winners)

##### CHAMPION #####
