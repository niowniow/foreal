from dash import Input, Output, callback, dcc, html

from ..app import WebportalPage


class IntroPage(WebportalPage):
    intro_text = """# Webportal
The foreal webportal is an easy and convenient way to explore foreal task graphs,
add interaction and debug the processing chain.
        """
    layout = html.Div(
        [
            dcc.Markdown(intro_text),
            dcc.Link("Go to Dashboard", href="/dashboard"),
        ]
    )
