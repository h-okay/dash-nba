from dash.dependencies import Input, Output, State
import dash
from dash import dash_table
from dash import Dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from app import app
import numpy as np
import glob
from dash_bootstrap_components import Button


def create_card(t_name):
    alt = " ".join(t_name.split("-")).title()
    image = app.get_asset_url(f"logos/NBA-{t_name}-Logo.png")
    return dbc.Card(
        [dbc.CardImg(src=image, top=True)],
        style={"display": "flex", "justify-content": "center", "align-items": "center"},
        outline=False,
        className="team_logo",
    )


def headshotCards(team):
    merged = pd.read_csv("data/base/merged.csv")
    merged = merged[(merged.TEAM == team) & (merged.SEASON_ID == "2021-22")]
    per = pd.read_csv("data/base/per.csv")
    per["NAME"] = per["FIRST_NAME"] + " " + per["LAST_NAME"]
    links = glob.glob(f"assets/top/{team}/*")
    files = pd.DataFrame({"LINK": links})
    files["NAME"] = files.LINK.apply(lambda x: x.split("\\")[1][:-4])
    names = (
        per[(per.TEAM == team) & (per.SEASON_ID == "2021-22")]
        .sort_values(by="PER", ascending=False)[:5]
        .reset_index(drop=True)[["NAME", "PER"]]
    )
    return names.merge(files, on="NAME", how="left").sort_values(
        by="PER", ascending=False
    )


def drawFigure(col_name, static, title, team, range=(650, 1300)):
    mlready = pd.read_csv("data/base/mlready.csv")
    mlready.SEASON = mlready.SEASON.apply(lambda x: int(x[:4]))
    mlready = mlready[mlready.TEAM == team]
    fig = px.area(
        mlready, x="SEASON", y=col_name, range_y=range, title=title
    ).update_layout(
        template="ggplot2",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        title_x=0.5,
        title_y=0.22,
        font=dict(size=18, color="white"),
        margin={"t": 0, "l": 0, "r": 0, "b": 0},
    )
    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.add_annotation(
        text=static,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"color": "white", "size": 48},
    )
    return html.Div(
        [
            dbc.Card(
                dbc.CardBody(
                    [
                        dcc.Graph(
                            figure=fig,
                            config={"displayModeBar": False},
                            className="Graph",
                        )
                    ]
                ),
                className="graph-card-background",
            ),
        ],
        className="graph-holder",
    )


def drawStats(team, player):
    per = pd.read_csv("data/base/per.csv")
    per["NAME"] = per["FIRST_NAME"] + " " + per["LAST_NAME"]
    per = per[(per.TEAM == team) & (per.SEASON_ID == "2021-22")]
    per["PPG"] = np.round(per["PTS"] / per["GP"], 2)
    per["RPG"] = np.round(per["REB"] / per["GP"], 2)
    per["APG"] = np.round(per["AST"] / per["GP"], 2)
    per = per[["NAME", "PPG", "PER"]].sort_values(by="PER", ascending=False).iloc[:5]
    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [html.P("PPG"), per[per.NAME == player].PPG.values[0]],
                        className="p-card-stats-card",
                    )
                ],
                className="p-card-stats",
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [html.P("PER"), per[per.NAME == player].PER.values[0]],
                        className="p-card-stats-card",
                    )
                ],
                className="p-card-stats",
            ),
        ],
        className="StatsRow",
        align=True,
        justify=True,
    )


def team_perf(team):
    mlready = pd.read_csv("data/base/mlready.csv")
    mlready = mlready[(mlready.TEAM == "Phoenix Suns") & (mlready.SEASON == "2021-22")]
    mlready["GP"] = mlready["W"] + mlready["L"]
    mlready["FG%"] = np.round(mlready["FGM"] / mlready["FGA"] * 100, 2)
    mlready["FG3%"] = np.round(mlready["FG3M"] / mlready["FG3A"] * 100, 2)
    mlready["FT%"] = np.round(mlready["FTM"] / mlready["FTA"] * 100, 2)
    mlready["PTS"] = np.round(mlready["PTS"] / mlready["GP"], 2)
    mlready["REB"] = np.round(mlready["REB"] / mlready["GP"], 2)
    mlready["AST"] = np.round(mlready["AST"] / mlready["GP"], 2)
    mlready["BLK"] = np.round(mlready["BLK"] / mlready["GP"], 2)
    mlready["STL"] = np.round(mlready["STL"] / mlready["GP"], 2)
    mlready["TOV"] = np.round(mlready["TOV"] / mlready["GP"], 2)
    mlready["PF"] = np.round(mlready["PF"] / mlready["GP"], 2)
    mlready["WIN%"] = mlready["WIN%"] * 100
    mlready = mlready[
        [
            "W",
            "L",
            "WIN%",
            "FG%",
            "FG3%",
            "FT%",
            "PTS",
            "REB",
            "AST",
            "BLK",
            "STL",
            "TOV",
            "PF",
        ]
    ]

    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [html.P("WIN"), mlready.W.values[0]],
                        className="t-card-stats-card",
                    )
                ],
                className="t-card-stats",
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [html.P("LOSE"), mlready.L.values[0]],
                        className="t-card-stats-card",
                    )
                ],
                className="t-card-stats",
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [html.P("WIN%"), mlready["WIN%"].values[0]],
                        className="t-card-stats-card",
                    )
                ],
                className="t-card-stats",
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [html.P("FG%"), mlready["FG%"].values[0]],
                        className="t-card-stats-card",
                    )
                ],
                className="t-card-stats",
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [html.P("FG3%"), mlready["FG3%"].values[0]],
                        className="t-card-stats-card",
                    )
                ],
                className="t-card-stats",
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [html.P("FT%"), mlready["FT%"].values[0]],
                        className="t-card-stats-card",
                    )
                ],
                className="t-card-stats",
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [html.P("PTS"), mlready.PTS.values[0]],
                        className="t-card-stats-card",
                    )
                ],
                className="t-card-stats",
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [html.P("REB"), mlready.REB.values[0]],
                        className="t-card-stats-card",
                    )
                ],
                className="t-card-stats",
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [html.P("AST"), mlready.AST.values[0]],
                        className="t-card-stats-card",
                    )
                ],
                className="t-card-stats",
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [html.P("BLK"), mlready.BLK.values[0]],
                        className="t-card-stats-card",
                    )
                ],
                className="t-card-stats",
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [html.P("STL"), mlready.STL.values[0]],
                        className="t-card-stats-card",
                    )
                ],
                className="t-card-stats",
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [html.P("TOV"), mlready.TOV.values[0]],
                        className="t-card-stats-card",
                    )
                ],
                className="t-card-stats",
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [html.P("PF"), mlready.PF.values[0]],
                        className="t-card-stats-card",
                    )
                ],
                className="t-card-stats",
            ),
        ],
        className="TeamStatsRow",
    )


def drawCard(team, no):
    hs = headshotCards(team)
    return dbc.Card(
        [
            html.P(
                [hs.NAME[no]],
                style={"font-family": "Inter, sans-serif", "margin-top": "10px"},
            ),
            html.Img(
                src=hs.LINK[no],
                width=188,
                height=137,
                style={"max-height": "100%", "max-width": "100%"},
            ),
            drawStats(team, hs.NAME[no]),
        ],
        className="shadow-card",
        id="log",
    )


def top_card(text, id):
    return [
        dbc.CardBody(
            [
                html.P(
                    text,
                    className="card-text",
                ),
            ],
            id=id,
        ),
    ]


def next_game(team):
    schedule = pd.read_csv("data/base/schedule.csv")
    schedule = schedule[(schedule.Away == team) | (schedule.Home == team)].sort_values(
        by="date"
    )
    next_game = schedule.iloc[0]
    first = next_game.Away
    second = next_game.Home
    return first, second


def matchup_info(team_):
    team, opponent = next_game(team_)



    team = 'Phoenix Suns'
    pd.set_option('display.max_columns', None)



    mlready = pd.read_csv("data/est/mlready.csv")
    standings = pd.read_csv("data/base/standingsCleaned.csv")
    st_tm = standings[
        (standings.TEAM == team) & (standings.SEASON == "2021-22")
    ].STREAK.values[0]
    st_op = standings[
        (standings.TEAM == opponent) & (standings.SEASON == "2021-22")
    ].STREAK.values[0]
    tm_per = mlready[(mlready.TEAM == team) & (mlready.SEASON == "2021-22")].PER.values[
        0
    ]
    op_per = mlready[
        (mlready.TEAM == opponent) & (mlready.SEASON == "2021-22")
    ].PER.values[0]
    tm_elo = int(
        mlready[(mlready.TEAM == team) & (mlready.SEASON == "2021-22")].ELO.values[0]
    )
    op_elo = int(
        mlready[(mlready.TEAM == opponent) & (mlready.SEASON == "2021-22")].ELO.values[
            0
        ]
    )
    return dbc.Card(
        [
            html.P(
                "Next Match",
                style={"font-family": "Inter, sans-serif", "margin-top": "10px"},
            ),
            html.H3(
                f"{team} vs. {opponent}",
                style={
                    "font-family": "Inter, sans-serif",
                    "margin-top": "10px",
                    "margin-bottom": "20px",
                    "color": "white",
                },
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    html.P("Win Probabilities"),
                                    f"67% - 33%",
                                ],  # need to calculate ##### !!!!
                                className="matchup-card-stats",
                            )
                        ],
                        id="matchup-info-col",
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [html.P("Average PERs"), f"{tm_per} - {op_per}"],
                                className="matchup-card-stats",
                            )
                        ],
                        id="matchup-info-col",
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [html.P("Team ELOs"), f"{tm_elo} - {op_elo}"],
                                className="matchup-card-stats",
                            )
                        ],
                        id="matchup-info-col",
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [html.P("Win Streaks"), f"{st_tm} - {st_op}"],
                                className="matchup-card-stats",
                            )
                        ],
                        id="matchup-info-col",
                    ),
                ],
                className="StatsRow",
                align=True,
                justify=True,
            ),
        ],
        className="shadow-card",
    )


def current_team_stats(team):
    standings = pd.read_csv("data/base/standingsCleaned.csv")
    st_tm = (
        standings[(standings.SEASON == "2021-22")]
        .sort_values(by="WIN%", ascending=False)
        .reset_index(drop=True)
    )
    st_tm["RANK"] = st_tm.index + 1
    st_tm = st_tm[["RANK", "TEAM", "W", "L", "LAST 10"]]
    idx = st_tm[st_tm.TEAM == team].index.values[0]
    if idx in list(range(0, 10)):
        st_tm = st_tm.iloc[:10]
    else:
        st_tm = st_tm.iloc[idx - 6 : idx + 4]
    return dash_table.DataTable(
        st_tm.to_dict("records"),
        [{"name": i, "id": i} for i in st_tm.columns],
        style_cell={
            "textAlign": "center",
            "background-color": "#242b44",
            "color": "white",
        },
        style_data_conditional=[
            {
                "if": {"row_index": idx, "column_id": "TEAM"},
                "backgroundColor": "#32a856",
            }
        ],
        style_header={
            "backgroundColor": "#242b44",
            "color": "white",
            "fontWeight": "bold",
            "textAlign": "left",
            "border": "1px solid black",
        },
        style_data={"border": "1px solid black"},
        id="st-table",
    )


def team_schedule(team):
    schedule = pd.read_csv("../data/base/schedule.csv")
    schedule = (
        schedule[(schedule.Away == team) | (schedule.Home == team)]
        .sort_values(by="date")
        .iloc[:10]
    )
    schedule = schedule.rename(columns={"date": "Date"})

    return dash_table.DataTable(
        schedule.to_dict("records"),
        [{"name": i, "id": i} for i in schedule.columns],
        style_cell={
            "textAlign": "center",
            "background-color": "#242b44",
            "color": "white",
        },
        style_header={
            "backgroundColor": "#242b44",
            "color": "white",
            "fontWeight": "bold",
            "textAlign": "left",
            "border": "1px solid black",
        },
        style_data={"border": "1px solid black"},
        id="sch-table",
    )


def player_perf(team):
    per = pd.read_csv("data/base/per.csv")
    per["NAME"] = per["FIRST_NAME"] + " " + per["LAST_NAME"]
    per = per[(per.TEAM == team) & (per.SEASON_ID == "2021-22")]
    per["PPG"] = np.round(per["PTS"] / per["GP"], 2)
    per["RPG"] = np.round(per["REB"] / per["GP"], 2)
    per["APG"] = np.round(per["AST"] / per["GP"], 2)
    per.drop(
        [
            "FIRST_NAME",
            "LAST_NAME",
            "P_ID",
            "SEASON_ID",
            "TEAM_ABBREVIATION",
            "TEAM",
            "MPG",
            "factor",
            "vop",
            "drbp",
            "uPER",
            "T_PACE",
            "L_PACE",
            "adjustment",
            "aPER",
            "PER",
        ],
        axis=1,
        inplace=True,
    )
    per = per.rename(columns={"FT_PCT": "FT%"})
    per = per[
        [
            "NAME",
            "AGE",
            "GP",
            "GS",
            "MIN",
            "FGM",
            "FGA",
            "FG%",
            "FG3M",
            "FG3A",
            "FG3%",
            "FTM",
            "FTA",
            "FT%",
            "OREB",
            "DREB",
            "REB",
            "AST",
            "STL",
            "BLK",
            "TOV",
            "PF",
            "PTS",
        ]
    ]
    per = per.sort_values(by="NAME")
    return dash_table.DataTable(
        per.to_dict("records"),
        [{"name": i, "id": i} for i in per.columns],
        style_cell={
            "textAlign": "center",
            "background-color": "#242b44",
            "color": "white",
        },
        style_header={
            "backgroundColor": "#242b44",
            "color": "white",
            "fontWeight": "bold",
            "textAlign": "left",
            "border": "1px solid black",
        },
        style_data={"border": "1px solid black"},
        id="sch-table",
    )


def performance_forecast_buttons(team):
    per = pd.read_csv("data/base/per.csv")
    per["NAME"] = per["FIRST_NAME"] + " " + per["LAST_NAME"]
    per = per[(per.TEAM == "Phoenix Suns") & (per.SEASON_ID == "2021-22")]
    per = per.reset_index()
    per = per.sort_values(by="NAME")
    players = per.NAME.unique()
    return [
        Button([player], id=f"btn-nclicks-{i+1}", n_clicks=0)
        for i, player in enumerate(players)
    ]


def get_button_count(team):
    per = pd.read_csv("data/base/per.csv")
    per["NAME"] = per["FIRST_NAME"] + " " + per["LAST_NAME"]
    per = per[(per.TEAM == "Phoenix Suns") & (per.SEASON_ID == "2021-22")]
    return len(sorted(per.NAME.unique()))
