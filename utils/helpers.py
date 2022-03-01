# import catboost
# import lightgbm as lgbm
# import h2o
import pandas as pd
# from h2o.automl import H2OAutoML
# from sklearn.ensemble import (
#     RandomForestRegressor,
#     ExtraTreesRegressor,
#     RandomForestClassifier,
# )
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import KFold, cross_validate
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler, OrdinalEncoder
# import xgboost as xgb
from tqdm import tqdm


# class AutoML:
#     def __init__(
#         self,
#         dataframe,
#         target,
#         test_size=0.25,
#         max_models=15,
#         save=False,
#         performance=True,
#     ):
#         h2o.init()
#         self.dataframe = dataframe
#         self.target = target
#         self.test_size = test_size
#         self.max_models = max_models
#         self.save = save
#         self.performance = performance
#
#     def prep_data(self):
#         label_vars = self.dataframe.drop(self.target, axis=1)
#         target_var = self.dataframe[self.target]
#         X_train, X_test, y_train, y_test = train_test_split(
#             label_vars, target_var, test_size=self.test_size, random_state=42
#         )
#         train = pd.concat([X_train, y_train], axis=1)
#         test = pd.concat([X_test, y_test], axis=1)
#
#         self.df = h2o.H2OFrame(self.train)
#         self.df_test = h2o.H2OFrame(self.test)
#
#         self.y = self.target
#         self.X = df.columns
#         self.X.remove(y)
#
#         return self.X, self.y, self_df, self.df_test
#
#     def train(self):
#         aml = H2OAutoML(max_models=self.max_models, seed=42)
#         aml.train(x=self.X, y=self.y, training_frame=self.df)
#         self.leader_model = aml.leader
#         return self.leader_model
#
#     def get_model(self):
#         self.X, self.y, self_df, self.df_test = self.prep_data()
#         self.leader_model = train()
#
#         if self.performance:
#             self.leader_model.model_performance(self.df_test)
#
#         if self.save:
#             self.leader_model.save_mojo(
#                 "C:/Users/Hakan/Desktop/NBA ML/prep/models/per_forecast_model.zip"
#             )


# class FastML:
#     def __init__(
#         self, objective, data, target, scale=False, ordinal=False, refit=False
#     ):
#         self.objective = objective
#         self.data = data
#         self.target = target
#         self.scale = scale
#         self.ordinal = ordinal
#         self.kfold = KFold(n_splits=5)
#         self.refit = refit
#
#         if self.objective not in ["regression", "classification"]:
#             raise f"{self.objective} is not valid."
#
#     @staticmethod
#     def grab_col_names(dataframe, cat_th=10, car_th=20):
#         cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
#         num_but_cat = [
#             col
#             for col in dataframe.columns
#             if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"
#         ]
#         cat_but_car = [
#             col
#             for col in dataframe.columns
#             if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"
#         ]
#         cat_cols = cat_cols + num_but_cat
#         cat_cols = [col for col in cat_cols if col not in cat_but_car]
#         num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
#         num_cols = [col for col in num_cols if col not in num_but_cat]
#         return cat_cols, num_cols, cat_but_car, num_but_cat
#
#     def classification_validate(self, model, X, y):
#         self.results = (
#             pd.DataFrame(
#                 cross_validate(
#                     model,
#                     X,
#                     y,
#                     cv=self.kfold,
#                     scoring=["neg_log_loss", "f1", "roc_auc"],
#                 )
#             )
#             .mean()
#             .to_frame()
#             .iloc[2:, :]
#             .rename(
#                 index={
#                     "test_f1": "F1-Score",
#                     "test_roc_auc": "ROC",
#                     "test_neg_log_loss": "Log-Loss",
#                 }
#             )
#         )
#         self.results = self.results.T
#         self.results["Log-Loss"] = self.results["Log-Loss"].apply(lambda x: -x)
#         return self.results[["Log-Loss", "F1-Score", "ROC"]]
#
#     def regression_validate(self, model, X, y):
#         self.results = (
#             pd.DataFrame(
#                 cross_validate(
#                     model,
#                     X,
#                     y,
#                     cv=self.kfold,
#                     scoring=["neg_root_mean_squared_error", "r2"],
#                 )
#             )
#             .mean()
#             .to_frame()
#             .iloc[2:, :]
#             .applymap(lambda x: -x)
#             .rename(index={"test_neg_root_mean_squared_error": "RMSE", "test_r2": "R2"})
#         )
#         self.results = self.results.T
#         self.results["R2"] = self.results["R2"].apply(lambda x: -x)
#         return self.results[["RMSE", "R2"]]
#
#     def select_models(self):
#         if self.objective == "classification":
#             self.models = [
#                 ("Catboost", catboost.CatBoostClassifier(random_state=42, silent=True)),
#                 (
#                     "XGBoost",
#                     xgb.XGBClassifier(
#                         random_state=42, verbosity=0, use_label_encoder=False
#                     ),
#                 ),
#                 ("LightGBM", lgbm.LGBMClassifier(random_state=42)),
#                 (
#                     "LogisticRegression",
#                     LogisticRegression(random_state=42, max_iter=100000),
#                 ),
#                 ("RandomForests", RandomForestClassifier(random_state=42)),
#                 ("KNN", KNeighborsClassifier()),
#             ]
#         if self.objective == "regression":
#             self.models = [
#                 ("Catboost", catboost.CatBoostRegressor(random_state=42, silent=True)),
#                 ("RandomForests", RandomForestRegressor(random_state=42)),
#                 ("ExtraTrees", ExtraTreesRegressor(random_state=42)),
#                 ("XGBoost", xgb.XGBRegressor(random_state=42)),
#                 ("LigthGBM", lgbm.LGBMRegressor(random_state=42)),
#             ]
#         return self.models
#
#     def data_prep(self):
#         self.X = self.data.drop(self.target, axis=1)
#         self.y = self.data[self.target]
#         if self.scale:
#             cat_cols, num_cols, cat_but_car, num_but_cat = self.grab_col_names(self.X)
#             ss = StandardScaler()
#             for col in num_cols:
#                 self.X[col] = ss.fit_transform(self.X[[col]])
#         if self.ordinal:
#             cat_cols, num_cols, cat_but_car, num_but_cat = self.grab_col_names(self.X)
#             for col in cat_cols:
#                 if self.X[col].dtype.name == "category":
#                     oe = OrdinalEncoder(
#                         categories=[self.X[col].dtype.categories.to_list()]
#                     )
#                     self.X[col] = oe.fit_transform(self.X[[col]])
#         self.X = pd.get_dummies(self.X, drop_first=True)
#         return self.X, self.y
#
#     def train_models(self, X, y):
#         if self.objective == "regression":
#             self.cv_result = pd.DataFrame()
#             pbar = tqdm(self.models, position=0, leave=True)
#             for model in pbar:
#                 pbar.set_description(f"{model[0]}")
#                 res = self.regression_validate(model[1], X, y)
#                 self.cv_result = pd.concat([self.cv_result, res])
#
#             self.cv_result.index = [
#                 "Catboost",
#                 "RandomForests",
#                 "ExtraTrees",
#                 "XGBoost",
#                 "LigthGBM",
#             ]
#
#             return self.cv_result.sort_values(
#                 by=["RMSE", "R2"], ascending=[True, False]
#             )
#
#         if self.objective == "classification":
#             self.cv_result = pd.DataFrame()
#             pbar = tqdm(self.models, position=0, leave=True)
#             for model in pbar:
#                 pbar.set_description(f"{model[0]}")
#                 res = self.classification_validate(model[1], X, y)
#                 self.cv_result = pd.concat([self.cv_result, res])
#
#             self.cv_result.index = [
#                 "Catboost",
#                 "XGBoost",
#                 "LightGBM",
#                 "LogisticRegression",
#                 "RandomForests",
#                 "KNN",
#             ]
#
#             return self.cv_result.sort_values(
#                 by=["Log-Loss", "F1-Score", "ROC"], ascending=[True, False, False]
#             )
#
#     def results(self):
#         if self.refit:
#             models = self.select_models()
#             X, y = self.data_prep()
#             top_models = self.train_models(X, y)
#             print(top_models)
#             self.best_model = top_models.index.values[0]
#             print(f"\nBest model was {self.best_model}. Refitting to return the model.")
#             for name, model in models:
#                 if name == self.best_model:
#                     fitted_model = model.fit(
#                         X,
#                         y,
#                     )
#             return fitted_model
#
#         else:
#             models = self.select_models()
#             X, y = self.data_prep()
#             return self.train_models(X, y)


def fix_team_names(row):
    if row in ["New Jersey Nets", "New York Nets", "Brooklyn"]:
        return "Brooklyn Nets"
    if row in [
        "Washington Bullets",
        "Baltimore Bullets",
        "Capital Bullets",
        "Washington",
    ]:
        return "Washington Wizards"
    if row in ["LA Clippers", "San Diego Clippers", "Buffalo Braves",
               "L.A. Clippers"]:
        return "Los Angeles Clippers"
    if row in ["Kansas City Kings", "Cincinnati Royals", "Kansas City",
               "Sacramento"]:
        return "Sacramento Kings"
    if row in ["Seattle SuperSonics", "Oklahoma", "Oklahoma City"]:
        return "Oklahoma City Thunder"
    if row in [
        "New Orleans/Oklahoma City Hornets",
        "New Orleans Hornets",
        "New Orleans",
    ]:
        return "New Orleans Pelicans"
    if row in ["Charlotte Bobcats", "Charlotte"]:
        return "Charlotte Hornets"
    if row in ["Vancouver Grizzlies", "Memphis"]:
        return "Memphis Grizzlies"
    if row in ["San Francisco Warriors", "Golden St."]:
        return "Golden State Warriors"
    if row in ["San Diego Rockets", "Houston"]:
        return "Houston Rockets"
    if row in ["New Orleans Jazz", "Utah"]:
        return "Utah Jazz"
    if row == "Boston":
        return "Boston Celtics"
    if row == "Atlanta":
        return "Atlanta Hawks"
    if row == "Chicago":
        return "Chicago Bulls"
    if row == "Cleveland":
        return "Cleveland Cavaliers"
    if row == "Dallas":
        return "Dallas Mavericks"
    if row == "Denver":
        return "Denver Nuggets"
    if row == "Detroit":
        return "Detroit Pistons"
    if row == "Indiana":
        return "Indiana Pacers"
    if row == "L.A. Lakers":
        return "Los Angeles Lakers"
    if row == "Miami":
        return "Miami Heat"
    if row == "Milwaukee":
        return "Milwaukee Bucks"
    if row == "Minnesota":
        return "Minnesota Timberwolves"
    if row == "New York":
        return "New York Knicks"
    if row == "Orlando":
        return "Orlando Magic"
    if row == "Philadelphia":
        return "Philadelphia 76ers"
    if row == "Phoenix":
        return "Phoenix Suns"
    if row == "Portland":
        return "Portland Trail Blazers"
    if row == "Toronto":
        return "Toronto Raptors"
    if row == "San Antonio":
        return "San Antonio Spurs"

    return row


def get_names(row):
    for i, v in enumerate(row):
        if v == "-":
            return row[:i].strip()
    return row


def get_matches():
    with open("prep/data/matches.pkl", "rb") as file:
        matches = pkl.load(file)

    for i in matches.values():
        cols = i.columns
        break

    starter = pd.DataFrame(columns=cols)
    for value in tqdm(matches.values()):
        starter = pd.concat([starter, value]).copy()
    starter["AWAY"] = starter.apply(lambda x: "@" in x["MATCHUP"], axis=1) * 1
    away_matches = starter[starter["AWAY"] == 1].copy()
    home_matches = starter[starter["AWAY"] == 0].copy()
    home_matches["WL_away"] = home_matches["WL"].apply(
        lambda x: "W" if x == "L" else "L"
    )
    home_matches["ABB_away"] = home_matches["MATCHUP"].apply(lambda x: x[-3:])
    concat_matches = home_matches.merge(
        away_matches,
        left_on=["GAME_ID", "WL_away", "ABB_away"],
        right_on=["GAME_ID", "WL", "TEAM_ABBREVIATION"],
    )
    concat_matches = concat_matches[
        [
            "SEASON_ID_x",
            "GAME_DATE_x",
            "TEAM_ID_x",
            "TEAM_NAME_x",
            "TEAM_ID_y",
            "TEAM_NAME_y",
            "WL_x",
            "WL_y",
            "MIN_x",
            "PTS_x",
            "PTS_y",
        ]
    ]

    team_ids = concat_matches["TEAM_ID_x"].unique()
    a = concat_matches[["TEAM_ID_x", "GAME_DATE_x"]]
    checkpoint = concat_matches[~a.duplicated()]  # 6 maç drop oldu.
    checkpoint = (
        checkpoint.sort_values("GAME_DATE_x").reset_index().drop("index",
                                                                 axis=1)
    )
    checkpoint.SEASON_ID_x = checkpoint.SEASON_ID_x.apply(
        lambda x: "2009-10" if x == "2009-010" else x
    )
    checkpoint.GAME_DATE_x = pd.to_datetime(checkpoint.GAME_DATE_x)
    checkpoint["MONTH"] = checkpoint.GAME_DATE_x.dt.month_name()
    checkpoint.drop(["TEAM_ID_x", "TEAM_ID_y", "MIN_x"], axis=1, inplace=True)
    return checkpoint
