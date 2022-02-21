import dash
from dash import Dash
import dash_bootstrap_components as dbc
from flask_caching import Cache


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
cache = Cache(
    app.server, config={"CACHE_TYPE": "filesystem", "CACHE_DIR": "cache-directory"}
)

server = app.server
app.config.suppress_callback_exceptions = True
