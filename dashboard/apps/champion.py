from dash.dependencies import Input, Output, State
import dash
from dash import Dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
from dashboard.app import app
import numpy as np
import glob
from dash import callback_context
from dash import dash_table
from dash_bootstrap_components import Button

from dashboard.helpers import fix_team_names

layout = dbc.Container(
    [
        dcc.Store(id="store-id-champ"),
        html.H2(["CHAMPIONSHIP PREDICTION"], id="champ_pred"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        Button(
                                            [year],
                                            id=f"champ-btn-{i + 1}",
                                            n_clicks=0,
                                            style={"font-size": "12.5px"},
                                        )
                                        for i, year in enumerate(
                                            ["2019", "2020", "2021", "2022"]
                                        )
                                    ],
                                    id="button-holder",
                                )
                            ],
                            id="champ-nav",
                        )
                    ]
                )
            ]
        ),
        dbc.Row(
            [dbc.Col([dbc.Card([], id="champ-placeholder", className="shadow-card")])]
        ),
    ]
)


@app.callback(
    Output("store-id-champ", "data"),
    [Input(f"champ-btn-{i+1}", "n_clicks") for i in range(4)],
)
def mvpnav(*args):
    trigger = callback_context.triggered[0]
    return trigger["prop_id"].split(".")[0].split("-")[-1]


@app.callback(
    Output("champ-placeholder", "children"), [Input("store-id-champ", "data")]
)
def draw_mvp_table(data):
    if data == "":
        data = 4
    else:
        data = int(data)
    selection = [(1, "2019"), (2, "2020"), (3, "2021"), (4, "2022")]
    sel = [val[1] for val in selection if val[0] == data][0]
    return html.Img(
        src=f"assets/champ_output/{sel}-Predictions-with-extra-features.png",
        style={"width": "100%", "heigth": "100%"},
    )
