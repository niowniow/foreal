import dash_bootstrap_components as dbc
from dash import Dash


# a way that we can have access to the same dash app
# from different files
def init(name=None, assets_folder="assets"):
    global app
    global config
    config = {}
    app = Dash(
        title=name,
        assets_folder=assets_folder,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
    )
