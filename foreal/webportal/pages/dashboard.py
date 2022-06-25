from dash import Input, Output, callback, dcc, html

from ..app import WebportalPage


class DashboardPage(WebportalPage):
    intro_text = """# Dashboard
The dashboard can be used to implement application-specific controls and plots.
This default dashboard does not contain any controls.
        """
    layout = html.Div(
        [
            dcc.Markdown(intro_text),
            dcc.Link("Go to Processing", href="/processing"),
        ]
    )
