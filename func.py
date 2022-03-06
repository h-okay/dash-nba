import dash
import dash_bootstrap_components as dbc
import glob
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
from dash import Dash
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash_bootstrap_components import Button
import os
import pathlib
import pickle as pkl


from app import app
from prep.scripts.classes import winProbability
from utils.helpers import fix_team_names

PATH = pathlib.Path(__file__)
DATA_PATH = PATH.joinpath("../prep/data").resolve()
EST_PATH = PATH.joinpath("../prep/estimations").resolve()


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
    merged = pd.read_csv(DATA_PATH.joinpath("merged.csv"))
    merged = merged[(merged.TEAM == team) & (merged.SEASON_ID == "2021-22")]
    per = pd.read_csv(DATA_PATH.joinpath("per.csv"))
    per["NAME"] = per["FIRST_NAME"] + " " + per["LAST_NAME"]
    per.dropna(inplace=True)
    links = glob.glob(f"assets/{team}/*")
    files = pd.DataFrame({"LINK": links})
    temp = [os.path.basename(val)[:-4] for val in files.LINK.to_list()]
    files["NAME"] = temp
    names = (
        per[(per.TEAM == team) & (per.SEASON_ID == "2021-22")]
        .sort_values(by="PER", ascending=False)[:5]
        .reset_index(drop=True)[["NAME", "PER"]]
    )
    return names.merge(files, on="NAME", how="left").sort_values(
        by="PER", ascending=False
    )


def drawFigure(col_name, static, title, team, range=(650, 1300)):
    mlready = pd.read_csv(DATA_PATH.joinpath("mlready.csv"))
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
    per = pd.read_csv(DATA_PATH.joinpath("per.csv"))
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
                        [
                            html.P("PER", style={"font-weight": "bold"}),
                            per[per.NAME == player].PER.values[0],
                        ],
                        className="p-card-stats-card",
                    )
                ],
                className="p-card-stats",
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [
                            html.P("PPG", style={"font-weight": "bold"}),
                            per[per.NAME == player].PPG.values[0],
                        ],
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
    mlready = pd.read_csv(DATA_PATH.joinpath("mlready.csv"))
    mlready = mlready[(mlready.TEAM == team) & (mlready.SEASON == "2021-22")]
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
                style={
                    "font-family": "Inter, sans-serif",
                    "margin-top": "10px",
                    "font-weight": "bold",
                },
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
                html.P(text, className="card-text", style={"font-weight": "bold"}),
            ],
            id=id,
        ),
    ]


def team_worth(team):
    temp = pd.read_csv(DATA_PATH.joinpath("salaries.csv"))
    temp = temp[(temp.TEAM == team) & (temp.YEAR == 2021)]
    total = temp.SALARY.sum()
    return "{:,}".format(total)


def next_game(team):
    schedule = pd.read_csv(DATA_PATH.joinpath("schedule.csv"))
    schedule.Away = schedule.Away.apply(fix_team_names)
    schedule.Home = schedule.Home.apply(fix_team_names)
    # sorted(schedule.Away.unique())
    schedule = schedule[(schedule.Away == team) | (schedule.Home == team)].sort_values(
        by="date"
    )
    next_game = schedule.iloc[0]
    first = next_game.Away
    second = next_game.Home
    return first, second


def matchup_info(team_):
    wp = winProbability(team_)
    wp.prep()
    winproba_df = wp.get_prediction()
    team, opponent = next_game(team_)
    mlready = pd.read_csv(DATA_PATH.joinpath("mlready.csv"))
    standings = pd.read_csv(DATA_PATH.joinpath("standingsCleaned.csv"))
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
            html.Div(
                html.H3(
                    f"{team} vs. {opponent}",
                    style={
                        "font-family": "Inter, sans-serif",
                        "margin-top": "10px",
                        "margin-bottom": "20px",
                        "color": "white",
                        "text-align": "center",
                    },
                ),
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    html.P("Win", style={"font-weight": "bold"}),
                                    f"{100 - int(winproba_df.iloc[0:, 1].values[0] * 100)}% - {int(winproba_df.iloc[0:, 1].values[0] * 100)}%",
                                ],
                                className="matchup-card-stats",
                            )
                        ],
                        id="matchup-info-col",
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    html.P(
                                        "Average PERs", style={"font-weight": "bold"}
                                    ),
                                    f"{tm_per} - {op_per}",
                                ],
                                className="matchup-card-stats",
                            )
                        ],
                        id="matchup-info-col",
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    html.P("Team ELOs", style={"font-weight": "bold"}),
                                    f"{tm_elo} - {op_elo}",
                                ],
                                className="matchup-card-stats",
                            )
                        ],
                        id="matchup-info-col",
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    html.P(
                                        "Win Streaks", style={"font-weight": "bold"}
                                    ),
                                    f"{st_tm} - {st_op}",
                                ],
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
    standings = pd.read_csv(DATA_PATH.joinpath("standingsCleaned.csv"))
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
        data=st_tm.to_dict("records"),
        columns=[{"name": i, "id": i} for i in st_tm.columns],
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
    schedule = pd.read_csv(DATA_PATH.joinpath("schedule.csv"))
    schedule.Away = schedule.Away.apply(fix_team_names)
    schedule.Home = schedule.Home.apply(fix_team_names)
    schedule = (
        schedule[(schedule.Away == team) | (schedule.Home == team)]
        .sort_values(by="date")
        .iloc[:10]
    )
    schedule = schedule.rename(columns={"date": "Date"})

    return dash_table.DataTable(
        data=schedule.to_dict("records"),
        columns=[{"name": i, "id": i} for i in schedule.columns],
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
    per = pd.read_csv(DATA_PATH.joinpath("per.csv"))
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
        data=per.to_dict("records"),
        columns=[{"name": i, "id": i} for i in per.columns],
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
    per = pd.read_csv(DATA_PATH.joinpath("per.csv"))
    per["NAME"] = per["FIRST_NAME"] + " " + per["LAST_NAME"]
    per = per[(per.TEAM == team) & (per.SEASON_ID == "2021-22")]
    per = per.reset_index()
    per = per.sort_values(by="NAME")
    players = per.NAME.unique()
    return html.Div(
        [
            Button(
                [player],
                id=f"btn-nclicks-{i + 1}",
                n_clicks=0,
                style={"font-size": "12.5px"},
            )
            for i, player in enumerate(players)
        ],
        id="button-holder",
    )


def worth_forecast_buttons(team):
    per = pd.read_csv(DATA_PATH.joinpath("per.csv"))
    per["NAME"] = per["FIRST_NAME"] + " " + per["LAST_NAME"]
    per = per[(per.TEAM == team) & (per.SEASON_ID == "2021-22")]
    per = per.reset_index()
    per = per.sort_values(by="NAME")
    players = per.NAME.unique()
    return html.Div(
        [
            Button(
                [player],
                id=f"worth-btn-nclicks-{i + 1}",
                n_clicks=0,
                style={"font-size": "12.5px"},
            )
            for i, player in enumerate(players)
        ],
        id="button-holder",
    )


def get_button_count(team):
    per = pd.read_csv(DATA_PATH.joinpath("per.csv"))
    per["NAME"] = per["FIRST_NAME"] + " " + per["LAST_NAME"]
    per = per[(per.TEAM == team) & (per.SEASON_ID == "2021-22")]
    return len(sorted(per.NAME.unique()))


def team_segmentation(team):
    segments = pd.read_csv(EST_PATH.joinpath("segmentation.csv"))
    segments["Segment"] = segments["Segment"].astype(str)
    fig = px.scatter_3d(
        segments,
        x="PER",
        y="MPG",
        z="AGE",
        hover_data=["TEAM", "POS"],
        hover_name="NAME",
        color="Segment",
        symbol=np.where(segments["TEAM"] == team, team, "Other"),
        symbol_map={team: "diamond", "Other": "cross"},
        size_max=10,
        opacity=0.5,
    ).update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(size=10, color="white"),
        margin={"t": 0, "l": 0, "r": 0, "b": 0},
        showlegend=False,
    )

    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    return dbc.Card(
        [
            dcc.Graph(
                figure=fig,
                config={"displayModeBar": False},
                className="Graph",
            )
        ],
        className="shadow-card",
    )


def elo_history(team):
    all_teams = pd.read_csv(DATA_PATH.joinpath("all_teams.csv")).get(
        ["id", "full_name"]
    )
    elo_ts = pd.read_csv(DATA_PATH.joinpath("save_elo_ts.csv"))
    elo_ts = (
        elo_ts.merge(all_teams, left_on=["TEAM_ID"], right_on="id")
        .drop(["id", "TEAM_ID"], axis=1)
        .sort_values(by="DATE")
    )
    elo_ts = elo_ts[elo_ts.full_name == team]
    max_elo = elo_ts.loc[elo_ts.ELO == elo_ts.ELO.max(), ["DATE", "ELO"]]
    min_elo = elo_ts.loc[elo_ts.ELO == elo_ts.ELO.min(), ["DATE", "ELO"]]
    fig = px.line(data_frame=elo_ts, x="DATE", y="ELO")
    fig.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(size=18, color="white"),
        margin={"t": 0, "l": 0, "r": 0, "b": 0},
        yaxis_title=None,
        xaxis_title=None,
        showlegend=False,
    )
    fig.update_traces(line_color="#03fcf8", line_width=2)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.add_trace(
        go.Scatter(
            x=max_elo["DATE"],
            y=max_elo["ELO"],
            name="HIGHEST ELO",
            marker_symbol="triangle-up",
            marker=dict(
                color="green",
                size=24,
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=min_elo["DATE"],
            y=min_elo["ELO"],
            name="LOWEST ELO",
            marker_symbol="triangle-down",
            marker=dict(
                color="red",
                size=24,
            ),
        )
    )
    return dbc.Card(
        [
            dcc.Graph(
                figure=fig,
                config={"displayModeBar": False},
                className="Graph",
            )
        ],
        className="shadow-card",
    )


def player_history(team):
    all_teams = pd.read_csv(DATA_PATH.joinpath("all_teams.csv")).get(
        ["id", "full_name"]
    )
    per = pd.read_csv(DATA_PATH.joinpath("per.csv"))
    per = per[(per.TEAM == team)]
    agg_per = per.groupby("SEASON_ID").PER.mean().reset_index()
    agg_per.SEASON_ID = agg_per.SEASON_ID.apply(lambda x: int(x[:4]))
    max_per = agg_per.loc[agg_per.PER == agg_per.PER.max(), ["SEASON_ID", "PER"]]
    min_per = agg_per.loc[agg_per.PER == agg_per.PER.min(), ["SEASON_ID", "PER"]]
    fig = px.line(data_frame=agg_per, x="SEASON_ID", y="PER", line_shape="hvh")
    fig.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(size=18, color="white"),
        margin={"t": 0, "l": 0, "r": 0, "b": 0},
        yaxis_title=None,
        xaxis_title=None,
        showlegend=False,
    )
    fig.update_traces(line_color="#03fcf8", line_width=2)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.add_trace(
        go.Scatter(
            x=max_per["SEASON_ID"],
            y=max_per["PER"],
            name="HIGHEST PER",
            marker_symbol="triangle-up",
            marker=dict(
                color="green",
                size=24,
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=min_per["SEASON_ID"],
            y=min_per["PER"],
            name="LOWEST PER",
            marker_symbol="triangle-down",
            marker=dict(
                color="red",
                size=24,
            ),
        )
    )
    return dbc.Card(
        [
            dcc.Graph(
                figure=fig,
                config={"displayModeBar": False},
                className="Graph",
            )
        ],
        className="shadow-card",
    )


def draw_kmeans():
    segments = pd.read_csv(EST_PATH.joinpath("segmentation.csv"))
    segments["Segment"] = segments["Segment"].astype(str)
    fig = px.scatter_3d(
        segments,
        x="PER",
        y="MPG",
        z="AGE",
        color="Segment",
        size_max=10,
        opacity=0.5,
    ).update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(size=10, color="white"),
        margin={"t": 0, "l": 0, "r": 0, "b": 0},
        showlegend=False,
    )

    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    return dbc.Card(
        [
            dcc.Graph(
                figure=fig,
                config={"displayModeBar": False},
                className="Graph",
            )
        ],
        className="shadow-card",
    )


def kmeans_table():
    segments = pd.read_csv(EST_PATH.joinpath("segmentation.csv"))
    segments["Segment"] = segments["Segment"].astype(str)
    agg_df = (
        segments.groupby("Segment")
        .agg(
            {
                "AGE": "mean",
                "PTS": "mean",
                "MPG": "mean",
                "PER": "mean",
                "SALARY": "mean",
                "NAME": "count",
            }
        )
        .rename(columns={"NAME": "COUNT"})
        .reindex(["Best", "Overperforming", "Average", "Underperforming", "Worst"])
        .round(2)
        .reset_index()
    )
    return dash_table.DataTable(
        data=agg_df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in agg_df.columns],
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


def segment_treemap():
    segments = pd.read_csv(EST_PATH.joinpath("segmentation.csv"))
    segments["Segment"] = segments["Segment"].astype(str)
    names = segments["NAME"].to_list()
    parents = segments["Segment"].to_list()
    fig = px.treemap(
        data_frame=segments, path=["Segment", "POS", "TEAM", "NAME"], values="PER"
    )
    fig.update_traces(root_color="rgba(0, 0, 0, 0)")
    fig.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(size=10, color="white"),
        margin={"t": 0, "l": 0, "r": 0, "b": 10},
        showlegend=False,
    )
    return dbc.Card(
        [
            dcc.Graph(
                figure=fig,
                config={"displayModeBar": False},
                className="Graph",
            )
        ],
        className="shadow-card",
    )


def news(team):
    path_ = DATA_PATH.joinpath("news.pkl")
    with open(path_, "rb") as file:
        news = pkl.load(file)
    try:
        return news[team][1]
    except (KeyError, IndexError):
        return f"{team} news needs to be updated..."


def draw_mvp_table():
    mvp = pd.read_csv("prep/estimations/mvps/2022_mvp.csv").round(4)
    return dash_table.DataTable(
        data=mvp.to_dict("records"),
        columns=[{"name": i, "id": i} for i in mvp.columns],
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
        id="mvp-table",
    )

def champ_photo():
    return html.Img(
        src=f"assets/champ_output/2022-Predictions-with-extra-features.png",
        style={"width": "100%", "heigth": "100%"},
    )

