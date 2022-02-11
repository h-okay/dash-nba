from dash.dependencies import Input, Output, State
import dash
from dash import Dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from app import app
from apps import home, phoenix_suns

from assets.style import SIDEBAR_STYLE, SIDEBAR_HIDEN, CONTENT_STYLE, CONTENT_STYLE1

navbar = dbc.NavbarSimple(
    children=[
        dbc.Button(
            html.Img(src="assets/sidebar-open-svgrepo-com.svg"),
            outline=False,
            color="primary",
            className="ml-1",
            id="btn_sidebar",
            style={"width": "100%", "height": "100%", "margin": "2 2 2 2"},
            size="sm",
        ),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Home", href="/"),
                dbc.DropdownMenuItem("Phoenix Suns", href="/phoenix_suns"),
            ],
            nav=True,
            in_navbar=True,
            label="Teams",
        ),
    ],
    color="rgba(5,28,45, 1)",
    dark=True,
    fluid=True,
    links_left=True,
    sticky="top",
    style={"height": "60px"},
)

sidebar = html.Div(
    [
        dbc.Nav(
            [
                html.P("Team Overview", className="display-4"),
                dbc.NavLink(
                    "Team",
                    href="#team",
                    id="page-1-link",
                    className="sidebar-links",
                    external_link=True,
                ),
                dbc.NavLink(
                    "Schedule",
                    href="#schedule",
                    id="page-2-link",
                    className="sidebar-links",
                    external_link=True,
                ),
                dbc.NavLink(
                    "Player Performances",
                    href="#p_performance",
                    id="page-3-link",
                    className="sidebar-links",
                    external_link=True,
                ),
                html.Hr(),
                html.P("Estimations", className="display-4"),
                dbc.NavLink(
                    "Future Performance",
                    href="#f_performance",
                    id="page-4-link",
                    className="sidebar-links",
                    external_link=True,
                ),
                dbc.NavLink(
                    "Current Worth",
                    href="/worth",
                    id="page-5-link",
                    className="sidebar-links",
                ),
                html.Hr(),
                html.P("Historic Data", className="display-4"),
                dbc.NavLink(
                    "Players",
                    href="/players",
                    id="page-6-link",
                    className="sidebar-links",
                ),
                dbc.NavLink(
                    "Team",
                    href="/teamhistoric",
                    id="page-7-link",
                    className="sidebar-links",
                ),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    id="sidebar",
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div(
    [
        dcc.Store(id="side_click"),
        dcc.Location(id="url"),
        navbar,
        sidebar,
        content,
    ],
)


@app.callback(
    [
        Output("sidebar", "style"),
        Output("page-content", "style"),
        Output("side_click", "data"),
    ],
    [Input("btn_sidebar", "n_clicks")],
    [
        State("side_click", "data"),
    ],
)
def toggle_sidebar(n, nclick):
    if n:
        if nclick == "SHOW":
            sidebar_style = SIDEBAR_HIDEN
            content_style = CONTENT_STYLE1
            cur_nclick = "HIDDEN"
        else:
            sidebar_style = SIDEBAR_STYLE
            content_style = CONTENT_STYLE
            cur_nclick = "SHOW"
    else:
        sidebar_style = SIDEBAR_STYLE
        content_style = CONTENT_STYLE
        cur_nclick = "SHOW"

    return sidebar_style, content_style, cur_nclick


@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 4)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        return True, False, False
    return [pathname == f"/page-{i}" for i in range(1, 4)]


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/home"]:
        return home.layout
    elif pathname == "/phoenix_suns":
        return phoenix_suns.layout


if __name__ == "__main__":
    app.run_server(host="127.0.0.1", debug=True)
