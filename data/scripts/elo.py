import pandas as pd
import pickle as pkl
import warnings
import numpy as np
from tqdm import tqdm

from pandas.core.common import SettingWithCopyWarning

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

all_players = pd.read_csv("data/all_players.csv")
all_teams = pd.read_csv("data/all_teams.csv")
merged = pd.read_csv("data/merged.csv")

with open("data/matches.pkl", "rb") as file:
    matches = pkl.load(file)

for i in matches.values():
    cols = i.columns
    break

########### MAÇLARI TEK SATIRA İNDİRGEME

starter = pd.DataFrame(columns=cols)  # Boş bir DataFrame oluştur.

# matches dictionarysindeki tüm takım maçlarını tek bir DataFrame'de topla.
for value in matches.values():
    starter = pd.concat([starter, value]).copy()

# Maçına baktığımız takımın AWAY olup olmadığını belirten bir değişken oluştur.
starter['AWAY'] = starter.apply(lambda x: '@' in x['MATCHUP'], axis=1) * 1
starter

# Takımları evinde ve deplasmanda oynayanlar olarak ayır.
away_matches = starter[starter['AWAY'] == 1].copy()
home_matches = starter[starter['AWAY'] == 0].copy()

# Home ve away match sayıları eşit değil.

away_matches
home_matches

# Karşı takımın kazanıp kazanmadığını belirten bir değişken ekle. (Merge ederken bazı sıkıntıları elemek için gerekli).
home_matches['WL_away'] = home_matches['WL'].apply(lambda x: 'W'
                                                   if x == 'L' else 'L')

# Karşı takımın kısaltmasını belirten bir değişken ekle. (Merge ederken faydası olabilir).
home_matches['ABB_away'] = home_matches['MATCHUP'].apply(lambda x: x[-3:])

# Home-Away maçlardaki tutarsızlıklara bak.
home_matches[~home_matches.GAME_ID.isin(away_matches.GAME_ID)].sort_values(
    'GAME_DATE')
away_matches[~away_matches.GAME_ID.isin(home_matches.GAME_ID)].sort_values(
    'GAME_DATE')

# Home-Away maçlardaki tutarsızlıklara bak. (Eğlence maçları)
home_matches[~home_matches.GAME_ID.isin(away_matches.GAME_ID)].sort_values(
    'GAME_DATE')
away_matches[~away_matches.GAME_ID.isin(home_matches.GAME_ID)].sort_values(
    'GAME_DATE')

# Belli bir tarihte birden fazla oyun oynanabilmekte.
home_matches.groupby('GAME_DATE')['GAME_ID'].nunique()[
    home_matches.groupby('GAME_DATE')['GAME_ID'].nunique() > 1]

# Home ve Away takım maçlarını birleştir.
concat_matches = home_matches.merge(
    away_matches,
    left_on=['GAME_ID', 'WL_away', 'ABB_away'],
    right_on=['GAME_ID', 'WL', 'TEAM_ABBREVIATION'])

###########

# NOT: Maçlar DataFrameinde temmuz ayında gerçekleşmiş fakat fikstürlerde bulunmayan maç bulunmakta.
# Tarih: 2009-07-10  , Ev Takımı: Dallas Mavericks, Rakip Takım: Milwaukee Bucks
# https://www.basketball-reference.com/teams/DAL/2010_games.html
# https://www.basketball-reference.com/teams/DAL/2009_games.html

# İşe yarayabileceğini düşündüğümüz değişkenleri seçelim.
concat_matches = concat_matches[[
    'SEASON_ID_x', 'GAME_DATE_x', 'TEAM_ID_x', 'TEAM_NAME_x', 'TEAM_ID_y',
    'TEAM_NAME_y', 'WL_x', 'WL_y', 'MIN_x', 'PTS_x', 'PTS_y'
]]
pd.set_option('display.width', 135)
concat_matches.head()
concat_matches['TEAM_ID_x'].nunique()  # Toplamda 30 takım bulunmakta.
concat_matches['TEAM_ID_y'].nunique()  # Toplamda 30 takım bulunmakta.

team_ids = concat_matches['TEAM_ID_x'].unique(
)  # Takım IDlerini tut. (Elo hesaplamasında kullanmak için)

elo_ratings = np.full(len(team_ids),
                      1400)  # Takımların başlangıç elolarını belirt.

a = concat_matches[[
    'TEAM_ID_x', 'GAME_DATE_x'
]]  # Duplicate satırları bulmak için eşsiz olabilecek değişkenleri seç.

concat_matches[a.duplicated()].sort_values(
    'GAME_DATE_x')  # Duplicate satırları bul. (6 maç)

# Duplicatelerin hepsi sezon dışı maçlar gibi gözükmekte. Maçların arasında bir çok sezon dışı maç bulunmakta.

## NOT: DUPLICATE SATIRLARIN SEBEBİNİ ÖĞREN. LİG DIŞI OLANLARI SALLA.

checkpoint = concat_matches[~a.duplicated()]  # 6 maç drop oldu.
checkpoint = checkpoint.sort_values('GAME_DATE_x').reset_index().drop('index',
                                                                      axis=1)

# Maçları tarihe göre sırala (ELO hesaplaması için gerekli) (Eğer reset index atılmazsa iterrows düzgün çalışmaz)

####### ELO HESAPLAMA


# Takımın maçı kazanma olasılığını hesapla
def expected_score(elo_a, elo_b):
    return 1 / (1 + 10**((elo_b - elo_a) / 400))


# Maçın sonucuna göre takımın yeni ELOsunu bul.
def new_rating(elo_a, score_a, expected_score_a):
    return elo_a + 32 * (score_a - expected_score_a)


elo_dict = dict(zip(
    team_ids,
    elo_ratings))  # Takım IDlerini ve elolarını sözlük aracılığıyla birleştir.

elo_date_team = pd.DataFrame(
    columns=['DATE','SEASON', 'TEAM_ID',
             'ELO'])  # Takımların tarihe göre elolarını kaydetmek için df


# Elo hesaplama döngüsü (10 Dakika Sürüyor)
for index, row in tqdm(checkpoint.iterrows()):
    season_id = row['SEASON_ID_x']
    game_date = row['GAME_DATE_x']
    home_id = row['TEAM_ID_x']
    home_elo = elo_dict[home_id]
    away_id = row['TEAM_ID_y']
    away_elo = elo_dict[away_id]
    exp_scr = expected_score(home_elo, away_elo)
    result = row['WL_x']

    if result == 'W':
        elo_dict[home_id] = new_rating(home_elo, 1, exp_scr)
        elo_dict[away_id] = new_rating(away_elo, 0, 1 - exp_scr)
    elif result == 'L':
        elo_dict[home_id] = new_rating(home_elo, 0, exp_scr)
        elo_dict[away_id] = new_rating(away_elo, 1, 1 - exp_scr)
    else:
        print(f'{season_id} season, {game_date} dated match is defected.')
        continue

    elo_date_team = elo_date_team.append(
        {
            'DATE': game_date,
            'SEASON': season_id,
            'TEAM_ID': home_id,
            'ELO': elo_dict[home_id]
        },
        ignore_index=True)
    elo_date_team = elo_date_team.append(
        {
            'DATE': game_date,
            'SEASON': season_id,
            'TEAM_ID': away_id,
            'ELO': elo_dict[away_id]
        },
        ignore_index=True)

elo_dict
asdf = elo_date_team.copy()
asdf.sort_values(by='DATE')
asdf.groupby(['DATE', 'SEASON', 'TEAM_ID']).ELO.value_counts()
asdf.groupby(['SEASON', 'TEAM_ID'])
c = asdf[asdf.groupby(['SEASON', 'TEAM_ID']).DATE.transform('max') == asdf.DATE]
c.to_csv('data/elos.csv', index=False)
# c[c.SEASON == '1983-84'].TEAM_ID.count()
import seaborn as sns
import matplotlib.pyplot as plt

# Rasgele bir takımın elo dağılımı
checkpoint[checkpoint['TEAM_ID_x'] == 1610612760]['TEAM_NAME_x'].unique(
)  # Takımın kullandığı tüm isimler

# Dağılımı çizdir
elos = elo_date_team[elo_date_team['TEAM_ID'] == 1610612760]['ELO']
sns.histplot(x=elos)
plt.show()

###################

team_stats = merged.groupby(["SEASON_ID", "TEAM"])[[
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
]].sum().reset_index()

#team_stats[team_stats.SEASON_ID == '1981-82'] # 81-82

playoffs = pd.read_csv('../data/playoffs.csv')
playoffs.Year = playoffs.Year.apply(lambda x: str(x - 1) + "-" + str(x)[-2:])
team_stats = team_stats[team_stats.SEASON_ID >= '1988-89']


def binary_generator(year, stage):
    global team_stats
    global playoffs
    season = str(year - 1) + "-" + str(year)[-2:]
    temp1 = playoffs[playoffs.Year == season]
    temp2 = temp1[temp1.Series == stage]
    a = temp2[['Year', 'Team1']].rename(columns={'Team1': 'Team'})
    b = temp2[['Year', 'Team2']].rename(columns={'Team2': 'Team'})
    c = pd.concat([a, b], axis=0, ignore_index=True)
    d = c.Team.to_list()
    team_stats.loc[(team_stats.SEASON_ID == season) &
                   (team_stats.TEAM.isin(d)), stage] = 1
    team_stats[stage] = team_stats[stage].fillna(0)


def champ_generator(year):
    global playoffs
    season = str(year - 1) + "-" + str(year)[-2:]
    chmp = playoffs.loc[(playoffs.Year == season) &
                        (playoffs.Series == 'Final'), 'Winner'].values[0]
    team_stats.loc[(team_stats.SEASON_ID == season) &
                   (team_stats.TEAM == chmp), 'Champion'] = 1
    team_stats['Champion'] = team_stats['Champion'].fillna(0)


for i in range(1989, 2022):
    binary_generator(i, 'Quarter')

team_stats

# for i in range(1989,2022):
#     champ_generator(i)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report

X = team_stats.drop('Playoff', axis=1)
X.drop(['SEASON_ID', 'TEAM'], axis=1, inplace=True)
y = team_stats['Playoff']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=42)

params = {
    'max_depth': np.arange(3, 15),
    'max_features': np.arange(2, len(X.columns))
}
rf = RandomForestClassifier(n_estimators=30)

rf_cv = RandomizedSearchCV(rf, param_distributions=params, cv=3, n_iter=30)

rf_cv.fit(X_train, y_train)

rf_cv.best_score_  # 0.717

imp = rf_cv.best_estimator_.feature_importances_

pd.DataFrame({
    'feature': X.columns,
    'importances': imp
}).sort_values('importances', ascending=False)

rf_best = rf_cv.best_estimator_
y_pred = rf_best.predict(X_test)
print(classification_report(y_test, y_pred))
proba = rf_best.predict_proba(X_test)

pd.DataFrame({'real': y_test, 'prob0': proba[:, 0], 'prob1': proba[:, 1]})
###############################################################################
for i in range(1989, 2022):
    binary_generator(i, 'Quarter')

X = team_stats.drop('Quarter', axis=1)
X.drop(['SEASON_ID', 'TEAM'], axis=1, inplace=True)
y = team_stats['Quarter']

params = {
    'max_depth': np.arange(3, 15),
    'max_features': np.arange(2, len(X.columns))
}
rf = RandomForestClassifier(n_estimators=500)

rf_cv = RandomizedSearchCV(rf, param_distributions=params, cv=3, n_iter=30)

rf_cv.fit(X, y)

rf_cv.best_score_

imp = rf_cv.best_estimator_.feature_importances_

pd.DataFrame({
    'feature': X.columns,
    'importances': imp
}).sort_values('importances', ascending=False)

proba = rf_best.predict_proba(X)

pd.DataFrame({'real': y, 'prob0': proba[:, 0], 'prob1': proba[:, 1]})
###############################################################################
for i in range(1989, 2022):
    binary_generator(i, 'Semi')

X = team_stats.drop('Semi', axis=1)
X.drop(['SEASON_ID', 'TEAM'], axis=1, inplace=True)
y = team_stats['Semi']

params = {
    'max_depth': np.arange(3, 15),
    'max_features': np.arange(2, len(X.columns))
}
rf = RandomForestClassifier(n_estimators=500)

rf_cv = RandomizedSearchCV(rf, param_distributions=params, cv=3, n_iter=30)

rf_cv.fit(X, y)

rf_cv.best_score_

imp = rf_cv.best_estimator_.feature_importances_

pd.DataFrame({
    'feature': X.columns,
    'importances': imp
}).sort_values('importances', ascending=False)

proba = rf_best.predict_proba(X)

pd.DataFrame({'real': y, 'prob0': proba[:, 0], 'prob1': proba[:, 1]})
###############################################################################
for i in range(1989, 2022):
    binary_generator(i, 'Final')

X = team_stats.drop('Final', axis=1)
X.drop(['SEASON_ID', 'TEAM'], axis=1, inplace=True)
y = team_stats['Final']

params = {
    'max_depth': np.arange(3, 15),
    'max_features': np.arange(2, len(X.columns))
}
rf = RandomForestClassifier(n_estimators=500)

rf_cv = RandomizedSearchCV(rf, param_distributions=params, cv=3, n_iter=30)

rf_cv.fit(X, y)

rf_cv.best_score_

imp = rf_cv.best_estimator_.feature_importances_

pd.DataFrame({
    'feature': X.columns,
    'importances': imp
}).sort_values('importances', ascending=False)

proba = rf_best.predict_proba(X)

pd.DataFrame({'real': y, 'prob0': proba[:, 0], 'prob1': proba[:, 1]})
