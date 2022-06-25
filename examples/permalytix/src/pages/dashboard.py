import dash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import pandas as pd
from click import launch
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from foreal.convenience import dict_update, to_datetime
from foreal.webportal import ForealControl

# # create a map of stations
# stations = pd.read_csv("./data/stationSelection.csv",names=['station','lat','lon'])
# stations['size'] = 10
# print(stations)
# import plotly.express as px

# stations_map_fig = px.scatter_mapbox(stations, lat="lat", lon="lon", hover_name="station",
#                         color_discrete_sequence=["blue"],zoom=12,size='size')
# stations_map_fig.update_layout(mapbox_style="open-street-map")
# stations_map_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})


class DatetimeControl(ForealControl):
    def __init__(self):
        start_time = "2021-06-18T01:26:00"
        end_time = "2021-06-18T01:28:00"
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        print(start_time.time())
        self.layout = html.Div(
            [
                dmc.Group(
                    children=[
                        dmc.TimeInput(
                            id="starttime-input",
                            label="Start time",
                            withSeconds=True,
                            clearable=True,
                            value=start_time,
                        ),
                        dmc.DateRangePicker(
                            id="date-range-picker",
                            label="Date Range",
                            # description="You can also provide a description",
                            minDate="20160101",
                            maxDate="20220101",
                            value=[start_time.date(), end_time.date()],
                            style={"width": 330},
                        ),
                        dmc.TimeInput(
                            id="endtime-input",
                            label="End time",
                            withSeconds=True,
                            clearable=True,
                            value=end_time,
                        ),
                    ]
                ),
                dmc.Group(
                    children=[
                        dcc.Input(
                            id="startdatetime-input",
                            type="text",
                            value="2022-06-05T11:20:00",
                        ),
                        dcc.Input(
                            id="enddatetime-input",
                            type="text",
                            value="2022-06-05T11:25:00",
                        ),
                    ]
                ),
            ]
        )
        self.elements_to_requests_mapping = {
            "startdatetime-input": ("value", "indexers.time.start"),
            "enddatetime-input": ("value", "indexers.time.stop"),
        }

    @callback(
        Output("startdatetime-input", "value"),
        Output("enddatetime-input", "value"),
        Input("starttime-input", "value"),
        Input("endtime-input", "value"),
        Input("date-range-picker", "value"),
    )
    def on_datetime_change(starttime, endtime, daterange):
        print(starttime, endtime, daterange)

        if daterange is None:
            raise PreventUpdate

        startdate = pd.to_datetime(daterange[0], utc=True)
        enddate = pd.to_datetime(daterange[1], utc=True)
        print(startdate, enddate)

        if starttime is not None:
            starttime = foreal.to_datetime(starttime)
            print(starttime, starttime.normalize())
            starttime = starttime - starttime.normalize()
            print(starttime)
            startdate = startdate + starttime

        if endtime is not None:
            endtime = foreal.to_datetime(endtime)
            print(endtime, endtime.normalize())
            endtime = endtime - endtime.normalize()
            print(endtime)
            enddate = enddate + endtime
        print(startdate, enddate)

        return (
            startdate.tz_localize(None).isoformat(),
            enddate.tz_localize(None).isoformat(),
        )


import dask
from benedict import benedict
from dask.core import flatten, get_dependencies
from holoviews.plotting.plotly.dash import to_dash

import foreal
from foreal.core import Node
from foreal.webportal import WebportalPage
from foreal.webportal.plotting import HoloViewPlot


class DashboardPage(WebportalPage):
    def __init__(self, app=None, taskgraphs=None):
        taskgraphs_names = list(taskgraphs.keys())
        taskgraphs_default_name = ""
        if taskgraphs_names:
            taskgraphs_default_name = taskgraphs_names[0]

        channel_layout = html.Div(
            [
                dmc.MultiSelect(
                    label="Channel select",
                    id="channel-multi-select",
                    searchable=True,
                    nothingFound="No channels found",
                    clearable=True,
                    value=["EHZ"],
                    data=[
                        {"value": "EHE", "label": "EHE"},
                        {"value": "EHZ", "label": "EHZ"},
                        {"value": "EHN", "label": "EHN"},
                    ],
                    style={"width": 400, "marginBottom": 10},
                ),
            ]
        )
        channel_control = ForealControl(
            channel_layout, {"channel-multi-select": ("value", "indexers.channel")}
        )

        # station_layout = html.Div([
        #     dmc.MultiSelect(
        #     label="Station select",
        #     id="station-multi-select",
        #     placeholder="Select one station",
        #     searchable=True,
        #     nothingFound="No stations found",
        #     clearable=True,
        #     maxSelectedValues=1,
        #     value=["MH36"],
        #     # data=[ {"value":f"ILL1{s+1}", "label":f"ILL1{s+1}"} for s in range(8)],
        #     data=[ {"value":f"MH36", "label":f"MH36"},{"value":f"MH38", "label":f"MH38"},{"value":f"111", "label":f"ILL13"}, {"value":f"106", "label":f"ILL16"}],
        #     style={"width": 400, "marginBottom": 10},
        # ),
        # ])
        # station_control = ForealControl(station_layout,{'station-multi-select':("value","indexers.station")})

        list_of_controls = [
            channel_control,
            # station_control,
            DatetimeControl(),
        ]
        self.list_of_controls = list_of_controls
        self.layout = html.Div(
            [
                # dcc.Store(id='request-memory'),
                dcc.Store(data="simple", id="graph-name"),
                html.H3("Dashboard"),
                # dmc.Select(label='task graph',data=taskgraphs_names, value=taskgraphs_default_name, id="taskgraphs-dropdown"),
                dmc.SimpleGrid(
                    cols=3,
                    children=[c.layout for c in list_of_controls],
                    id="control-div",
                ),
                dmc.Button(id="update-button", children="Update"),
                dmc.LoadingOverlay(
                    dmc.Text("Nothing to show yet", size="xl"), id="dashboard_plot"
                ),
                dbc.Toast(
                    [
                        html.P(
                            f"In this demo, we restricted to timeframe for which you can load data. Please choose a shorter time frame of max 5 minutes.",
                            className="mb-0",
                        )
                    ],
                    id="timeframe-toast",
                    header="Choose a shorter timeframe",
                    icon="primary",
                    dismissable=True,
                    duration=10000,
                    is_open=False,
                    style={"position": "fixed", "top": 66, "right": 10, "width": 350},
                ),
                # html.Div("Nothing to show yet",id="dashboard_plot"),
            ]
        )
        super().__init__(app=app, taskgraphs=taskgraphs)

    def callbacks(self, app, taskgraphs):
        @callback(
            # Output("control-div", "children"),
            Output("graph-name", "data"),
            Input("taskgraphs-dropdown", "value"),
            # prevent_initial_call=True,
        )
        def select_graph(graphname):
            return graphname

        hvp = HoloViewPlot()

        args = (
            Output("dashboard_plot", "children"),
            Output("timeframe-toast", "is_open"),
            Input("update-button", "n_clicks"),
            State("graph-name", "data"),
        )
        args_list = list(args)
        for control in self.list_of_controls:
            for id, mapping in control.elements_to_requests_mapping.items():
                args_list += [State(id, mapping[0])]
        args = tuple(args_list)

        @callback(
            *args,
            prevent_initial_call=True,
        )
        def on_update_graphs(n_clicks, graph_name, *args):
            ctx = dash.callback_context

            if not ctx.triggered:
                raise PreventUpdate

            input_id = ctx.triggered[0]["prop_id"].split(".")[0]

            request = benedict({"self": {}})
            for state in ctx.states:
                state_name = state.split(".")[0]
                request["self"][state_name] = ctx.states[state]
            for control in self.list_of_controls:
                request = control.control(request)
            if "self" in request:
                del request["self"]

            if input_id == "update-button":
                diff = to_datetime(request["indexers.time.stop"]) - to_datetime(
                    request["indexers.time.start"]
                )
                print("doff", diff)
                if diff > pd.to_timedelta(5, "minutes"):
                    return html.Div(children="Please, choose a shorter timeframe"), True
                configured_collection = foreal.core.configuration(
                    taskgraphs[graph_name]["default"], request, optimize_graph=False
                )
                x = dask.compute(configured_collection)
                hv_plots = hvp.forward(x, request)
                components = []
                for plot in hv_plots:
                    components += to_dash(app, plot)[0]

                return html.Div(children=components), False

            raise PreventUpdate
