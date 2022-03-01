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

from dashboard.helpers import fix_team_names
from dashboard.func import draw_kmeans, kmeans_table, segment_treemap


layout = dbc.Container(
    [
        html.H2(["TREE MAP"], id="clustering1"),
        html.Hr(),
        dbc.Row([dbc.Col([segment_treemap()])]),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.H2(["HIERARCHICAL CLUSTERING"], id="clustering2"),
        html.Hr(),
        dbc.Row([dbc.Col([dbc.Card([kmeans_table()], id="table-card5")])]),
        dbc.Row(
            [
                dbc.Col([draw_kmeans()]),
            ]
        ),
    ]
)
