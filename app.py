import dash
import dash_bootstrap_components as dbc
from dash import Dash

external_stylesheets = [dbc.themes.LUX]

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, maximum-scale=1",
        }
    ],
)

server = app.server
app.config.suppress_callback_exceptions = True
