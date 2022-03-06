import dash_bootstrap_components as dbc
import pandas as pd
from dash import callback_context
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import os
import pathlib
from func import draw_mvp_table

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../prep/data").resolve()
EST_PATH = PATH.joinpath("../prep/estimations").resolve()
from app import app, server

from dash_bootstrap_components import Button


layout = dbc.Container(
    [
        dcc.Store(id="store-id-mvp"),
        html.H2(["MVP PREDICTION"], id="mvp_pred"),
        html.Hr(),
        dbc.Row(
            [dbc.Col([dbc.Card([draw_mvp_table()], id="mvp-placeholder", className="shadow-card")])]
        ),
    ]
)


