import dash_bootstrap_components as dbc
from dash import callback_context
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import os
import pathlib
from func import champ_photo

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../prep/data").resolve()
EST_PATH = PATH.joinpath("../prep/estimations").resolve()
from app import app, server

from dash_bootstrap_components import Button


layout = dbc.Container(
    [
        dcc.Store(id="store-id-champ"),
        html.H2(["CHAMPIONSHIP PREDICTION"], id="champ_pred"),
        html.Hr(),
        dbc.Row(
            [dbc.Col([dbc.Card([champ_photo()], id="champ-placeholder", className="shadow-card")])]
        ),
    ]
)

