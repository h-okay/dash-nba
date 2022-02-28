from dash.dependencies import Input, Output, State
import dash
from dash import Dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from app import app
from apps import (
    atlanta_hawks,
    boston_celtics,
    brooklyn_nets,
    charlotte_hornets,
    chicago_bulls,
    cleveland_cavaliers,
    dallas_mavericks,
    denver_nuggets,
    detroit_pistons,
    golden_state_warriors,
    home,
    houston_rockets,
    indiana_pacers,
    los_angeles_clippers,
    los_angeles_lakers,
    memphis_grizzlies,
    miami_heat,
    milwaukee_bucks,
    minnesota_timberwolves,
    new_orleans_pelicans,
    new_york_knicks,
    oklahoma_city_thunder,
    orlando_magic,
    philadelphia_76ers,
    phoenix_suns,
    portland_trail_blazers,
    sacramento_kings,
    san_antonio_spurs,
    toronto_raptors,
    utah_jazz,
    washington_wizards,
    kmeans,
    mvp,
    champion,
)

from assets.style import SIDEBAR_STYLE, SIDEBAR_HIDEN, CONTENT_STYLE, CONTENT_STYLE1

navbar = dbc.NavbarSimple(
    children=[
        dbc.Button(
            html.Img(src="assets/menu.svg", width=24, height=24),
            outline=False,
            color="secondary",
            className="ml-1",
            id="btn_sidebar",
            style={
                "width": "100%",
                "height": "100%",
                "margin": "2 2 2 2",
                "background-color": "transparent",
                "border-color": "transparent",
            },
            size="sm",
        ),
        dbc.Button(
            html.Img(src="assets/home.svg", width=24, height=24),
            outline=False,
            color="secondary",
            className="ml-1",
            id="btn_home",
            style={
                "width": "100%",
                "height": "100%",
                "margin": "2 2 2 2",
                "background-color": "transparent",
                "border-color": "transparent",
            },
            size="sm",
            href="/home",
            external_link=True,
        ),
        dbc.Button(
            html.Img(src="assets/kmeans.svg", width=24, height=24),
            outline=False,
            color="secondary",
            className="ml-1",
            id="segmentate_button",
            style={
                "width": "100%",
                "height": "100%",
                "margin": "2 2 2 2",
                "background-color": "transparent",
                "border-color": "transparent",
            },
            size="sm",
            href="/kmeans",
            external_link=True,
        ),
        dbc.Button(
            html.Img(src="assets/champion.svg", width=24, height=24),
            outline=False,
            color="secondary",
            className="ml-1",
            id="championship_button",
            style={
                "width": "100%",
                "height": "100%",
                "margin": "2 2 2 2",
                "background-color": "transparent",
                "border-color": "transparent",
            },
            size="sm",
            href="/champion",
            external_link=True,
        ),
        dbc.Button(
            html.Img(src="assets/mvp.svg", width=24, height=24),
            outline=False,
            color="secondary",
            className="ml-1",
            id="mvp_button",
            style={
                "width": "100%",
                "height": "100%",
                "margin": "2 2 2 2",
                "background-color": "transparent",
                "border-color": "transparent",
            },
            size="sm",
            href="/mvp",
            external_link=True,
        ),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem(
                    "Atlanta Hawks",
                    href="/atlanta_hawks",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Boston Celtics",
                    href="/boston_celtics",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Brooklyn Nets",
                    href="/brooklyn_nets",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Charlotte Hornets",
                    href="/charlotte_hornets",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Chicago Bulls",
                    href="/chicago_bulls",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Cleveland Cavaliers",
                    href="/cleveland_cavaliers",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Dallas Mavericks",
                    href="/dallas_mavericks",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Denver Nuggets",
                    href="/denver_nuggets",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Detroit Pistons",
                    href="/detroit_piston",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Golden State Warriors",
                    href="/golden_state_warriors",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Houston Rockets",
                    href="/houston_rockets",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Indiana Pacers",
                    href="/indiana_pacers",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Los Angeles Clippers",
                    href="/los_angeles_clippers",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Los Angeles Lakers",
                    href="/los_angeles_lakers",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Memphis Grizzlies",
                    href="/memphis_grizzlies",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Miami Heat", href="/miami_heat", style={"font-size": "12.5px"}
                ),
                dbc.DropdownMenuItem(
                    "Milwaukee Bucks",
                    href="/milwaukee_bucks",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Minnesota Timberwolves",
                    href="/minnesota_timberwolves",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "New Orleans Pelicans",
                    href="/new_orleans_pelicans",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "New York Knicks",
                    href="/new_york_knicks",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Oklahoma City Thunder",
                    href="/oklahoma_city_thunder",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Orlando Magic",
                    href="/orlando_magic",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Philadelphia 76ers",
                    href="/philadelphia_76ers",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Phoenix Suns", href="/phoenix_suns", style={"font-size": "12.5px"}
                ),
                dbc.DropdownMenuItem(
                    "Portland Trail Blazers",
                    href="/portland_trail_blazers",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Sacramento Kings",
                    href="/sacramento_kings",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "San Antonio Spurs",
                    href="/san_antonio_spurs",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Toronto Raptors",
                    href="/toronto_raptors",
                    style={"font-size": "12.5px"},
                ),
                dbc.DropdownMenuItem(
                    "Utah Jazz", href="/utah_jazz", style={"font-size": "12.5px"}
                ),
                dbc.DropdownMenuItem(
                    "Washington Wizards",
                    href="/washington_wizards",
                    style={"font-size": "12.5px"},
                ),
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
                    href="#p_worth",
                    id="page-5-link",
                    className="sidebar-links",
                    external_link=True,
                ),
                dbc.NavLink(
                    "Player Segmentation",
                    href="#p_segment",
                    id="page-6-link",
                    className="sidebar-links",
                    external_link=True,
                ),
                html.Hr(),
                html.P("Historic Data", className="display-4"),
                dbc.NavLink(
                    "Players",
                    href="#player_history",
                    id="page-7-link",
                    className="sidebar-links",
                    external_link=True,
                ),
                dbc.NavLink(
                    "Team",
                    href="#elo_history",
                    id="page-8-link",
                    className="sidebar-links",
                    external_link=True,
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
        sidebar_style = SIDEBAR_HIDEN
        content_style = CONTENT_STYLE1
        cur_nclick = "HIDDEN"

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
    elif pathname == "/kmeans":
        return kmeans.layout
    elif pathname == "/champion":
        return champion.layout
    elif pathname == "/allstar":
        return allstar.layout
    elif pathname == "/mvp":
        return mvp.layout
    elif pathname == "/atlanta_hawks":
        return atlanta_hawks.layout
    elif pathname == "/boston_celtics":
        return boston_celtics.layout
    elif pathname == "/brooklyn_nets":
        return brooklyn_nets.layout
    elif pathname == "/charlotte_hornets":
        return charlotte_hornets.layout
    elif pathname == "/chicago_bulls":
        return chicago_bulls.layout
    elif pathname == "/cleveland_cavaliers":
        return cleveland_cavaliers.layout
    elif pathname == "/dallas_mavericks":
        return dallas_mavericks.layout
    elif pathname == "/denver_nuggets":
        return denver_nuggets.layout
    elif pathname == "/detroit_piston":
        return detroit_piston.layout
    elif pathname == "/golden_state_warriors":
        return golden_state_warriors.layout
    elif pathname == "/houston_rockets":
        return houston_rockets.layout
    elif pathname == "/indiana_pacers":
        return indiana_pacers.layout
    elif pathname == "/los_angeles_clippers":
        return los_angeles_clippers.layout
    elif pathname == "/los_angeles_lakers":
        return los_angeles_lakers.layout
    elif pathname == "/memphis_grizzlies":
        return memphis_grizzlies.layout
    elif pathname == "/miami_heat":
        return miami_heat.layout
    elif pathname == "/milwaukee_bucks":
        return milwaukee_bucks.layout
    elif pathname == "/minnesota_timberwolves":
        return minnesota_timberwolves.layout
    elif pathname == "/new_orleans_pelicans":
        return new_orleans_pelicans.layout
    elif pathname == "/new_york_knicks":
        return new_york_knicks.layout
    elif pathname == "/oklahoma_city_thunder":
        return oklahoma_city_thunder.layout
    elif pathname == "/orlando_magic":
        return orlando_magic.layout
    elif pathname == "/philadelphia_76ers":
        return philadelphia_76ers.layout
    elif pathname == "/phoenix_suns":
        return phoenix_suns.layout
    elif pathname == "/portland_trail_blazers":
        return portland_trail_blazers.layout
    elif pathname == "/sacramento_kings":
        return sacramento_kings.layout
    elif pathname == "/san_antonio_spurs":
        return san_antonio_spurs.layout
    elif pathname == "/toronto_raptors":
        return toronto_raptors.layout
    elif pathname == "/utah_jazz":
        return utah_jazz.layout
    elif pathname == "/washington_wizards":
        return washington_wizards.layout


if __name__ == "__main__":
    app.run_server(host="127.0.0.1")
