from dash.dependencies import Input, Output, State
import dash
from dash import Dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from dashboard.app import app
import numpy as np
import glob
from dash import callback_context
from dash import dash_table

from dashboard.helpers import fix_team_names
from prep.scripts.classes import winProbability

from dashboard.func import (
    headshotCards,
    drawFigure,
    drawStats,
    team_perf,
    drawCard,
    top_card,
    matchup_info,
    current_team_stats,
    team_schedule,
    player_perf,
    performance_forecast_buttons,
    get_button_count,
    team_worth,
    worth_forecast_buttons,
    team_segmentation,
    elo_history,
    player_history,
    # team_segment_table
)

hs = headshotCards("Oklahoma City Thunder")
n_buttons = get_button_count("Oklahoma City Thunder")
team_ = "Oklahoma City Thunder"

layout = dbc.Container(
    [
        dcc.Store(id="store-id24"),
        dcc.Store(id="worth-id24"),
        html.H2(["TEAM"], id="team"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.Card(
                            top_card(
                                f"Team Worth: {team_worth('Oklahoma City Thunder')} $",
                                "worth-cardbody",
                            ),
                            color="success",
                            inverse=True,
                        ),
                        id="worth-card1",
                    ),
                    id="worth",
                    width=6,
                    align="center",
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.Card(
                            top_card(
                                "Devin Booker and Chris Paul named NBA All-Stars",
                                "news-card-body",
                            ),
                            color="danger",
                            inverse=True,
                        ),
                        id="worth-card2",
                    ),
                    id="worth",
                    width=6,
                    align="center",
                ),
            ],
            align="center",
            id="worth-row",
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.CardBody(
                            [
                                drawCard("Oklahoma City Thunder", 0),
                                drawCard("Oklahoma City Thunder", 1),
                                drawCard("Oklahoma City Thunder", 2),
                                drawCard("Oklahoma City Thunder", 3),
                                drawCard("Oklahoma City Thunder", 4),
                            ],
                            id="PlayerCard",
                        )
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        matchup_info("Oklahoma City Thunder"),
                        dbc.Card(
                            current_team_stats("Oklahoma City Thunder"), id="table-card"
                        ),
                    ],
                    width=6,
                ),
            ],
            id="second-row",
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.H2(["SCHEDULE"], id="schedule"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [dbc.Card(team_schedule("Oklahoma City Thunder"), id="table-card2")]
                )
            ]
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.H2(["PLAYER PERFORMANCES"], id="p_performance"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [dbc.Card(player_perf("Oklahoma City Thunder"), id="table-card3")]
                )
            ]
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.H2(["PERFORMANCE FORECAST"], id="f_performance"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            performance_forecast_buttons("Oklahoma City Thunder"),
                            id="forecast-card",
                            className="shadow-card",
                        )
                    ],
                    width=12,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(children=[], width=3, id="placeholder93"),
                dbc.Col(children=[], width=9, id="placeholder94"),
            ]
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.H2(["PLAYER WORTH"], id="p_worth"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            worth_forecast_buttons("Oklahoma City Thunder"),
                            id="salary-nav",
                            className="shadow-card",
                        )
                    ]
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col([], id="placeholder95", width=3),
                dbc.Col([], id="placeholder96", width=9),
            ],
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.H2(["PLAYER SEGMENTATION"], id="p_segment"),
        html.Hr(),
        dbc.Row([dbc.Col([team_segmentation("Oklahoma City Thunder")])]),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.H2(["PLAYER HISTORY"], id="player_history"),
        html.Hr(),
        dbc.Row([dbc.Col([player_history("Oklahoma City Thunder")])]),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.H2(["TEAM HISTORY"], id="elo_history"),
        html.Hr(),
        dbc.Row([dbc.Col([elo_history("Oklahoma City Thunder")])]),
    ],
    className="team-page-container",
)


@app.callback(
    Output("store-id24", "data"),
    [Input(f"btn-nclicks-{i+1}", "n_clicks") for i in range(n_buttons)],
)
def perfnav(*args):
    trigger = callback_context.triggered[0]
    return trigger["prop_id"].split(".")[0].split("-")[-1]


@app.callback(
    Output("worth-id24", "data"),
    [Input(f"worth-btn-nclicks-{i+1}", "n_clicks") for i in range(n_buttons)],
)
def salarynav(*args):
    trigger = callback_context.triggered[0]
    return trigger["prop_id"].split(".")[0].split("-")[-1]


@app.callback(Output("placeholder93", "children"), [Input("store-id24", "data")])
def player_performance(data, team=team_):
    if data == "":
        data = 1
    else:
        data = int(data)
    merged = pd.read_csv("prep/data/merged.csv")
    merged = merged[(merged.TEAM == team) & (merged.SEASON_ID == "2021-22")]
    per = pd.read_csv("prep/data/per.csv")
    per["NAME"] = per["FIRST_NAME"] + " " + per["LAST_NAME"]
    links = glob.glob(f"dashboard/assets/{team}/*")
    files = pd.DataFrame({"LINK": links})
    files["NAME"] = files.LINK.apply(lambda x: x.split("\\")[1][:-4])
    names = per[(per.TEAM == team) & (per.SEASON_ID == "2021-22")].reset_index(
        drop=True
    )[["NAME", "PER"]]
    hs = names.merge(files, on="NAME", how="left").sort_values(by="NAME")
    per = per[(per.TEAM == team) & (per.SEASON_ID == "2021-22")]
    per["PPG"] = np.round(per["PTS"] / per["GP"], 2)
    per["RPG"] = np.round(per["REB"] / per["GP"], 2)
    per["APG"] = np.round(per["AST"] / per["GP"], 2)
    per["FG%"] = np.round(per["FG%"] * 100, 2)
    per["FG3%"] = np.round(per["FG3%"] * 100, 2)
    per["FT_PCT"] = np.round(per["FT_PCT"] * 100, 2)
    per["STL"] = np.round(per["STL"] / per["GP"], 2)
    per["BLK"] = np.round(per["BLK"] / per["GP"], 2)
    per["TOV"] = np.round(per["TOV"] / per["GP"], 2)
    per["PF"] = np.round(per["PF"] / per["GP"], 2)
    per = per[
        [
            "NAME",
            "PPG",
            "PER",
            "RPG",
            "APG",
            "MPG",
            "FG%",
            "FG3%",
            "FT_PCT",
            "STL",
            "BLK",
            "TOV",
            "PF",
        ]
    ].sort_values(by="NAME")
    per = per.reset_index(drop=True)
    hs = hs.reset_index(drop=True)
    forecast = pd.read_csv("prep/estimations/perf_forecast.csv")
    forecast.columns
    p_select = hs.NAME[data - 1]
    forecast = forecast[(forecast.NAME == p_select)]
    forecast["SEASON"] = forecast.SEASON_ID.apply(lambda x: int(x[:4]) + 1)
    played = forecast[["TEAM", "SEASON"]]
    played = played[~played.duplicated()]
    played = played.groupby("TEAM").SEASON.min().reset_index().sort_values(by="SEASON")

    return dbc.Card(
        [
            html.P(
                [hs.NAME[data - 1]],
                style={
                    "font-family": "Inter, sans-serif",
                    "margin-top": "10px",
                    "font-weight": "bold",
                },
            ),
            html.Img(
                src=hs.LINK[data - 1][10:].replace('\\', '/'),
                width=188,
                height=137,
                style={"max-height": "100%", "max-width": "100%"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    html.P("PER", style={"font-weight": "bold"}),
                                    per[per.NAME == hs.NAME[data - 1]].PER.values[0],
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
                                    per[per.NAME == hs.NAME[data - 1]].PPG.values[0],
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
                                    html.P("RPG", style={"font-weight": "bold"}),
                                    per[per.NAME == hs.NAME[data - 1]].RPG.values[0],
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
                                    html.P("APG", style={"font-weight": "bold"}),
                                    per[per.NAME == hs.NAME[data - 1]].APG.values[0],
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
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    html.P("FG%", style={"font-weight": "bold"}),
                                    per[per.NAME == hs.NAME[data - 1]]["FG%"].values[0],
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
                                    html.P("FG3%", style={"font-weight": "bold"}),
                                    per[per.NAME == hs.NAME[data - 1]]["FG3%"].values[
                                        0
                                    ],
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
                                    html.P("FT%", style={"font-weight": "bold"}),
                                    per[per.NAME == hs.NAME[data - 1]]["FT_PCT"].values[
                                        0
                                    ],
                                ],
                                className="p-card-stats-card",
                            )
                        ],
                        className="p-card-stats",
                    ),
                ],
                className="StatsRow",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    html.P("STL", style={"font-weight": "bold"}),
                                    per[per.NAME == hs.NAME[data - 1]]["STL"].values[0],
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
                                    html.P("BLK", style={"font-weight": "bold"}),
                                    per[per.NAME == hs.NAME[data - 1]]["BLK"].values[0],
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
                                    html.P("TOV", style={"font-weight": "bold"}),
                                    per[per.NAME == hs.NAME[data - 1]]["TOV"].values[0],
                                ],
                                className="p-card-stats-card",
                            )
                        ],
                        className="p-card-stats",
                    ),
                ],
                className="StatsRow",
            ),
        ],
        className="shadow-card",
    )


@app.callback(Output("placeholder94", "children"), [Input("store-id24", "data")])
def per_forecast(data, team=team_):
    if data == "":
        data = 1
    else:
        data = int(data)
    per = pd.read_csv("prep/data/per.csv")
    per["NAME"] = per["FIRST_NAME"] + " " + per["LAST_NAME"]
    names = per[(per.TEAM == team) & (per.SEASON_ID == "2021-22")].reset_index(
        drop=True
    )[["NAME", "PER"]]
    hs = names.sort_values(by="NAME")
    per = per[(per.TEAM == team) & (per.SEASON_ID == "2021-22")]
    per["PPG"] = np.round(per["PTS"] / per["GP"], 2)
    per["RPG"] = np.round(per["REB"] / per["GP"], 2)
    per["APG"] = np.round(per["AST"] / per["GP"], 2)
    per = per[["NAME", "PPG", "PER", "RPG", "APG", "MPG"]].sort_values(by="NAME")
    per = per.reset_index(drop=True)
    hs = hs.reset_index(drop=True)
    forecast = pd.read_csv("prep/estimations/perf_forecast.csv")
    p_select = hs.NAME[data - 1]
    forecast = forecast[(forecast.NAME == p_select)]
    forecast["SEASON"] = forecast.SEASON_ID.apply(lambda x: int(x[:4]) + 1)
    lst = [
        (row[1].SEASON, row[1].PER)
        if row[1].SEASON_ID != "2021-22"
        else (row[1].SEASON, np.round(row[1].PRED, 2))
        for row in forecast.iterrows()
    ]
    g = pd.DataFrame(lst)
    g.columns = ["year", "per"]
    g = np.round(g.groupby("year").per.mean(), 2).reset_index()
    permean = g.per.mean()
    permax = g.per.max()

    fig = px.bar(
        data_frame=g,
        x="year",
        y="per",
        text="per",
        range_y=(0, permax + permean),
        height=397,
    ).update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(size=18, color="white"),
        margin={"t": 0, "l": 0, "r": 0, "b": 0},
        yaxis_title=None,
        xaxis_title=None,
        showlegend=False,
    )
    fig.update_xaxes(dtick="date", showgrid=False)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_traces(
        textfont_size=16, textangle=0, textposition="outside", cliponaxis=False
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


@app.callback(Output("placeholder95", "children"), [Input("worth-id24", "data")])
def player_performance(data, team=team_):
    if data == "":
        data = 1
    else:
        data = int(data)
    merged = pd.read_csv("prep/data/merged.csv")
    merged = merged[(merged.TEAM == team) & (merged.SEASON_ID == "2021-22")]
    per = pd.read_csv("prep/data/per.csv")
    per["NAME"] = per["FIRST_NAME"] + " " + per["LAST_NAME"]
    links = glob.glob(f"dashboard/assets/{team}/*")
    files = pd.DataFrame({"LINK": links})
    files["NAME"] = files.LINK.apply(lambda x: x.split("\\")[1][:-4])
    names = per[(per.TEAM == team) & (per.SEASON_ID == "2021-22")].reset_index(
        drop=True
    )[["NAME", "PER"]]
    hs = names.merge(files, on="NAME", how="left").sort_values(by="NAME")
    per = per[(per.TEAM == team) & (per.SEASON_ID == "2021-22")]
    per["PPG"] = np.round(per["PTS"] / per["GP"], 2)
    per["RPG"] = np.round(per["REB"] / per["GP"], 2)
    per["APG"] = np.round(per["AST"] / per["GP"], 2)
    per["FG%"] = np.round(per["FG%"] * 100, 2)
    per["FG3%"] = np.round(per["FG3%"] * 100, 2)
    per["FT_PCT"] = np.round(per["FT_PCT"] * 100, 2)
    per["STL"] = np.round(per["STL"] / per["GP"], 2)
    per["BLK"] = np.round(per["BLK"] / per["GP"], 2)
    per["TOV"] = np.round(per["TOV"] / per["GP"], 2)
    per["PF"] = np.round(per["PF"] / per["GP"], 2)
    per = per[
        [
            "NAME",
            "PPG",
            "PER",
            "RPG",
            "APG",
            "MPG",
            "FG%",
            "FG3%",
            "FT_PCT",
            "STL",
            "BLK",
            "TOV",
            "PF",
        ]
    ].sort_values(by="NAME")
    per = per.reset_index(drop=True)
    hs = hs.reset_index(drop=True)
    forecast = pd.read_csv("prep/estimations/perf_forecast.csv")
    p_select = hs.NAME[data - 1]
    forecast = forecast[(forecast.NAME == p_select)]
    forecast["SEASON"] = forecast.SEASON_ID.apply(lambda x: int(x[:4]) + 1)
    played = forecast[["TEAM", "SEASON"]]
    played = played[~played.duplicated()]
    played = played.groupby("TEAM").SEASON.min().reset_index().sort_values(by="SEASON")

    return dbc.Card(
        [
            html.P(
                [hs.NAME[data - 1]],
                style={
                    "font-family": "Inter, sans-serif",
                    "margin-top": "10px",
                    "font-weight": "bold",
                },
            ),
            html.Img(
                src=hs.LINK[data - 1][10:].replace('\\', '/'),
                width=188,
                height=137,
                style={"max-height": "100%", "max-width": "100%"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    html.P("PER", style={"font-weight": "bold"}),
                                    per[per.NAME == hs.NAME[data - 1]].PER.values[0],
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
                                    per[per.NAME == hs.NAME[data - 1]].PPG.values[0],
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
                                    html.P("RPG", style={"font-weight": "bold"}),
                                    per[per.NAME == hs.NAME[data - 1]].RPG.values[0],
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
                                    html.P("APG", style={"font-weight": "bold"}),
                                    per[per.NAME == hs.NAME[data - 1]].APG.values[0],
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
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    html.P("FG%", style={"font-weight": "bold"}),
                                    per[per.NAME == hs.NAME[data - 1]]["FG%"].values[0],
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
                                    html.P("FG3%", style={"font-weight": "bold"}),
                                    per[per.NAME == hs.NAME[data - 1]]["FG3%"].values[
                                        0
                                    ],
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
                                    html.P("FT%", style={"font-weight": "bold"}),
                                    per[per.NAME == hs.NAME[data - 1]]["FT_PCT"].values[
                                        0
                                    ],
                                ],
                                className="p-card-stats-card",
                            )
                        ],
                        className="p-card-stats",
                    ),
                ],
                className="StatsRow",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    html.P("STL", style={"font-weight": "bold"}),
                                    per[per.NAME == hs.NAME[data - 1]]["STL"].values[0],
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
                                    html.P("BLK", style={"font-weight": "bold"}),
                                    per[per.NAME == hs.NAME[data - 1]]["BLK"].values[0],
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
                                    html.P("TOV", style={"font-weight": "bold"}),
                                    per[per.NAME == hs.NAME[data - 1]]["TOV"].values[0],
                                ],
                                className="p-card-stats-card",
                            )
                        ],
                        className="p-card-stats",
                    ),
                ],
                className="StatsRow",
            ),
        ],
        className="shadow-card",
    )


@app.callback(Output("placeholder96", "children"), [Input("worth-id24", "data")])
def player_worth(data, team=team_):
    if data == "":
        data = 1
    else:
        data = int(data)
    per = pd.read_csv("prep/data/per.csv")
    per["NAME"] = per["FIRST_NAME"] + " " + per["LAST_NAME"]
    names = per[(per.TEAM == team) & (per.SEASON_ID == "2021-22")].reset_index(
        drop=True
    )[["NAME", "PER"]]
    hs = names.sort_values(by="NAME").reset_index(drop=True)
    p_select = hs.NAME[data - 1]
    salaries = pd.read_csv("prep/data/salaries.csv")
    salaries.TEAM = salaries.TEAM.apply(fix_team_names)
    salaries = salaries[(salaries.NAME == p_select)]
    salmax = salaries.SALARY.max()
    salmean = salaries.SALARY.mean()

    fig = px.bar(
        data_frame=salaries,
        x="YEAR",
        y="SALARY",
        color="TEAM",
        text="SALARY",
        range_y=(0, salmax + salmean),
        height=397,
    ).update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(size=18, color="white"),
        margin={"t": 0, "l": 0, "r": 0, "b": 0},
        yaxis_title=None,
        xaxis_title=None,
        showlegend=False,
    )

    fig.update_xaxes(dtick="date", showgrid=False)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_traces(
        textfont_size=16, textangle=0, textposition="outside", cliponaxis=False
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