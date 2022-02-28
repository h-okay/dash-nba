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
from app import app
import numpy as np
import glob
from dash import callback_context
from dash import dash_table
from dash_bootstrap_components import Button

from dashboard.helpers import fix_team_names

layout = dbc.Container(
    [
        dcc.Store(id="store-id-mvp"),
        html.H2(["MVP PREDICTION"], id="mvp_pred"),
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
                                            id=f"mvp-btn-{i + 1}",
                                            n_clicks=0,
                                            style={"font-size": "12.5px"},
                                        )
                                        for i, year in enumerate(
                                            ["2018", "2019", "2020", "2021", "2022"]
                                        )
                                    ],
                                    id="button-holder",
                                )
                            ],
                            id="mvp-nav",
                        )
                    ]
                )
            ]
        ),
        dbc.Row(
            [dbc.Col([dbc.Card([], id="mvp-placeholder", className="shadow-card")])]
        ),
    ]
)


@app.callback(
    Output("store-id-mvp", "data"),
    [Input(f"mvp-btn-{i+1}", "n_clicks") for i in range(5)],
)
def mvpnav(*args):
    trigger = callback_context.triggered[0]
    return trigger["prop_id"].split(".")[0].split("-")[-1]


@app.callback(Output("mvp-placeholder", "children"), [Input("store-id-mvp", "data")])
def draw_mvp_table(data):
    if data == "":
        data = 5
    else:
        data = int(data)
    selection = [(1, "2018"), (2, "2019"), (3, "2020"), (4, "2021"), (5, "2022")]
    sel = [val[1] for val in selection if val[0] == data][0]
    mvp = pd.read_csv(f"../prep/estimations/mvps/{sel}_mvp.csv").round(4)
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
