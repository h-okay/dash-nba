# import dash_bootstrap_components as dbc
# from dash import dcc
# from dash import html
# from dash.dependencies import Input, Output
# from dash import callback_context
# from func import layout_generator
#
# PATH = pathlib.Path(__file__)
DATA_PATH = PATH.joinpath("../prep/data").resolve()
EST_PATH = PATH.joinpath("../prep/estimations").resolve()
pd.read_csv("prep/data/merged.csv")
# hs, n_buttons, team_, layout = layout_generator(trigger["prop_id"].split(".")[0].split("-")[-1])
#
# app.layout = dbc.Container(
#     [dbc.DropdownMenu(children=[
#         dbc.DropdownMenuItem(
#             "Atlanta Hawks",
#             href="/atlanta_hawks",
#             style={"font-size": "12.5px"},
#             id='Atlanta Hawks'
#         ),
#         dbc.DropdownMenuItem(
#             "Boston Celtics",
#             href="/boston_celtics",
#             style={"font-size": "12.5px"},
#             id='Boston Celtics')
#     ],
#         nav=True,
#         in_navbar=True,
#         label="Teams",
#         id="nav_dropdown"
#     ),
#         html.Div([], id='my-output')])
#
# @app.callback(
#     Output("my-output", "children"),
#     [Input("Atlanta Hawks", "n_clicks"),
#      Input("Boston Celtics", "n_clicks")],
# )
# def perfnav(*args):
#     trigger = callback_context.triggered[0]
#     layout_generator(trigger["prop_id"].split(".")[0].split("-")[-1])
#     return
#
#
# if __name__ == '__main__':
#     app.run_server(debug=True)
