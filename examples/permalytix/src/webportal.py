import base64

from foreal.webportal import create_webportal

from .pages.dashboard import DashboardPage
from .taskgraphs.simple import get_taskgraphs

# the taskgraphs are the important part. They will be loaded by the webportal
# and made available to interact with.
taskgraphs = {"simple": get_taskgraphs()}

# just some customizations
branding = {
    "page": "https://www.foreal.io/",
    "name": "permalytix",
}

# create the defaul webapp with foReal branding and our taskgraph
app = create_webportal(taskgraphs, branding=branding, dashboard=DashboardPage)

if __name__ == "__main__":
    app.run_server(debug=False, port=8008, dev_tools_props_check=False)
