import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, callback, dcc, html, no_update

from . import shared


class WebportalPage:
    def __init__(self, app=None, taskgraphs=None):
        self.app = app
        self.taskgraphs = taskgraphs

        if self.app is not None and hasattr(self, "callbacks"):
            self.callbacks(self.app, self.taskgraphs)


def create_webportal(
    taskgraphs,
    branding=None,
    introduction=None,
    processing=None,
    dashboard=None,
    introduction_kwargs=None,
    processing_kwargs=None,
    dashboard_kwargs=None,
):
    """Create a webportal based on a foreal task graph.

    Args:
        taskgraphs (foreal graph): The task graph which will be used in the portal
        branding (dict, optional): A dict containing the keys 'logo' (a link to an image or base64 encoded image),'page' (a link to a webpage),'name' (the name of the brand). Defaults to None.
        introduction (dash layout, optional): A dash layout for the intro page. If None, the default intro page will be shown. Defaults to None.
        processing (dash layout, optional): A dash layout for the processing page. If None, the default processing page will be shown. Defaults to None.
        dashboard (dash layout, optional): A dash layout for the dashboard page. If None, the default dashboard page will be shown. Defaults to None.

    Returns:
        dash app: The webportal as a dash app. It can be started with `app.run_server()`
    """

    # provide a base branding if nothing was given or components are missing
    from .utils import logo

    foreal_branding = {"logo": logo, "page": "foreal.io", "name": "foreal"}
    if branding is not None:
        foreal_branding.update(branding)
    branding = foreal_branding

    # this call will create our dash app in a shared global context which is avaiable
    # to all subpages
    shared.init(branding["name"])

    app = shared.app
    server = app.server

    # shared.config.update(kwargs)

    shared.config["taskgraphs"] = taskgraphs

    # go to default pages if nothing was provided
    if introduction is None:
        from .pages.introduction import IntroPage as introduction
    if processing is None:
        from .pages.processing import ProcessingPage as processing
    if dashboard is None:
        from .pages.dashboard import DashboardPage as dashboard

    if introduction_kwargs is None:
        introduction_kwargs = {}
    if processing_kwargs is None:
        processing_kwargs = {}
    if dashboard_kwargs is None:
        dashboard_kwargs = {}

    shared.config["introduction"] = introduction(
        app=app, taskgraphs=taskgraphs, **introduction_kwargs
    )
    shared.config["processing"] = processing(
        app=app, taskgraphs=taskgraphs, **processing_kwargs
    )
    shared.config["dashboard"] = dashboard(
        app=app, taskgraphs=taskgraphs, **dashboard_kwargs
    )

    nav_items = [
        dbc.NavItem(dbc.NavLink("Info", href="/")),
        dbc.NavItem(dbc.NavLink("Dashboard", href="/dashboard")),
        dbc.NavItem(dbc.NavLink("Processing", href="/processing")),
    ]

    dropdown = dbc.DropdownMenu(
        children=[
            dbc.DropdownMenuItem("Download Data", id="download-data"),
            dbc.DropdownMenuItem("Download SVG", id="download-svg"),
            dbc.DropdownMenuItem(divider=True),
            dbc.DropdownMenuItem("About", href=branding["page"]),
        ],
        nav=True,
        in_navbar=True,
        label="Menu",
    )

    @callback(
        Output("auto-toast", "is_open"),
        [Input("download-data", "n_clicks"), Input("download-svg", "n_clicks")],
    )
    def open_toast(n, m):
        if n or m:
            return True
        return no_update

    # this example that adds a logo to the navbar brand
    logo = dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    # Use row and col to control vertical alignment of logo / brand
                    dbc.Row(
                        [
                            dbc.Col(html.Img(src=branding["logo"], height="30px")),
                            dbc.Col(
                                dbc.NavbarBrand(branding["name"], className="ms-2")
                            ),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    href=branding["page"],
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                dbc.Collapse(
                    dbc.Nav(
                        nav_items + [dropdown],
                        className="ms-auto",
                        navbar=True,
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ],
        ),
        color="#F5F5FF",
        dark=False,
        className="mb-5",
    )

    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            logo,
            html.Div(id="page-content"),
            dbc.Toast(
                [
                    html.P(
                        f"This feature has been disabled in this demo. For more information please contact us via {branding['page']}",
                        className="mb-0",
                    )
                ],
                id="auto-toast",
                header="Feature is disabled",
                icon="primary",
                dismissable=True,
                duration=10000,
                is_open=False,
                style={"position": "fixed", "top": 66, "right": 10, "width": 350},
            ),
        ]
    )

    app.validation_layout = html.Div(
        [
            shared.config["processing"].layout,
            shared.config["dashboard"].layout,
            shared.config["introduction"].layout,
            app.layout,
        ]
    )

    # we use a callback to toggle the collapse on small screens
    def toggle_navbar_collapse(n, is_open):
        if n:
            return not is_open
        return is_open

    app.callback(
        Output(f"navbar-collapse", "is_open"),
        [Input(f"navbar-toggler", "n_clicks")],
        [State(f"navbar-collapse", "is_open")],
    )(toggle_navbar_collapse)

    return app


@callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return shared.config["introduction"].layout
    elif pathname == "/dashboard":
        return shared.config["dashboard"].layout
    elif pathname == "/processing":
        return shared.config["processing"].layout
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        dbc.Container(
            [
                html.H1("404: Not found", className="text-danger"),
                html.P(f"The pathname {pathname} was not recognised..."),
                html.Hr(className="my-2"),
                html.P(
                    dbc.Button("Go Home", color="primary", href="/"), className="lead"
                ),
            ],
            fluid=True,
            className="py-3",
        ),
        className="p-3 bg-light rounded-3",
    )
