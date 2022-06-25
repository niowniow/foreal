# from dash import dcc, html, Input, Output, callback

# layout = html.Div(
#     [
#         html.H3("Page 1"),
#         # dcc.Dropdown(
#         #     {f'Page 1 - {i}': f'{i}' for i in ['New York City', 'Montreal', 'Los Angeles']},
#         #     id='page-1-dropdown'
#         # ),
#         html.Div(id="page-1-display-value"),
#         dcc.Link("Go to Page 2", href="/page2"),
#     ]
# )


# @callback(Output("page-1-display-value", "children"), Input("page-1-dropdown", "value"))
# def display_value(value):
#     return f"You have selected {value}"


import ast
import json
import math
from select import select

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import dash_mantine_components as dmc
import dask
import holoviews as hv
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from dash import callback, dcc, html
from dash.dependencies import Input, Output, State
from dask.core import flatten, get_dependencies
from dask.delayed import Delayed
from flask import request
from holoviews.plotting.plotly.dash import to_dash
from matplotlib.pyplot import text

import foreal
from foreal.apps.annotations import requests_from_file, to_formatted_json
from foreal.core.graph import base_name

cyto.load_extra_layouts()

from .. import shared

text_request = """{
    "indexers": {
       "time":{"start":"2022-06-05T11:20:00","stop":"2022-06-05T11:40:00"},
    },
}"""


from foreal.convenience import dict_update

from ..app import WebportalPage


class ProcessingPage(WebportalPage):
    def __init__(self, app=None, taskgraphs=None):
        graph_names = []
        for app_name in taskgraphs:
            if "." in app_name:
                raise RuntimeError(
                    "The taskgraph collection name should not contain `.`"
                )

            d = taskgraphs[app_name]
            names = [
                {"value": f"{app_name}.{k}", "label": f"{app_name}.{k}"}
                for k in d.keys()
                if k != "instances"
            ]
            graph_names += names

        app_name, taskgraph_name = graph_names[0]["label"].split(".")[:2]
        taskgraph = taskgraphs[app_name][taskgraph_name]

        # taskgraph = foreal.core.configuration(taskgraph, r, optimize_graph=False)
        dsk, dsk_keys = dask.base._extract_graph_and_keys([taskgraph])

        work = list(set(flatten(dsk_keys)))

        dsk_dict = dsk.to_dict()
        # dsk_dict = dsk
        nodes = []
        edges = []

        roots = ["#" + k for k in work]
        while work:
            new_work = {}
            for k in work:
                nodes.append({"data": {"id": k, "label": k}})

                current_deps = get_dependencies(dsk_dict, k, as_list=True)

                for dep in current_deps:
                    edges.append({"data": {"source": dep, "target": k}})

                    if dep not in work:
                        new_work[dep] = True

            work = new_work

        elements = nodes + edges

        self.layout = html.Div(
            [
                html.H2("1. Select a task graph"),
                dcc.Dropdown(
                    options=graph_names,
                    value=graph_names[0]["value"],
                    id="graphs-dropdown",
                ),
                html.H2("2. Select output to visualize"),
                dcc.Markdown(
                    "Click on a node to select which output should be plotted. If no node is selected, the output of the last node is plotted and the configuration of all nodes can be viewed by clicking show default configuration"
                ),
                cyto.Cytoscape(
                    id="cytoscape-taskgraph",
                    elements=elements,
                    userZoomingEnabled=False,
                    style={"width": "100%", "height": "200px"},
                    layout={
                        "name": "dagre",
                        "rankDir": "LR",
                        "roots": ",".join(roots),
                    },
                ),
                dmc.Button(
                    id="show-config-button", children="Show default configuration"
                ),
                html.H2("3. Set configuration"),
                # dcc.Dropdown(options=catalog_names, value=0, id="configs-dropdown"),
                dcc.Textarea(
                    id="textarea-state-example",
                    value=text_request,
                    style={"width": "100%", "height": 200},
                ),
                # dcc.Input(
                #     id="request_target",
                #     type="text",
                #     placeholder="all",
                #     value='all',
                # ),
                # html.Button("4. Compute", id="textarea-state-example-button", n_clicks=0),
                html.H2("4. Run Job"),
                # html.Div(
                #     [
                #         html.P(id="paragraph_id"),
                #         html.Progress(id="progress_bar"),
                #     ]
                # ),
                # html.Button(id="compute_button_id", children="Compute"),
                dmc.Button(id="compute_button_id", children="Compute"),
                # html.Button(id="cancel_button_id", children="Cancel Running Job!"),
                html.H2("Results"),
                html.Details(
                    [
                        html.Summary("Details of computed output"),
                        cyto.Cytoscape(
                            id="cytoscape-computed",
                            elements=elements[:1],
                            userZoomingEnabled=False,
                            style={"width": "100%", "height": "200px"},
                            layout={
                                "name": "dagre",
                                "rankDir": "LR",
                                "roots": ",".join(roots),
                            },
                        ),
                        dcc.Markdown("", id="config-computed"),
                    ]
                ),
                dmc.LoadingOverlay(
                    dmc.Text("No plots yet", size="m"), id="loading-plot"
                ),
                dmc.Drawer(
                    children=[],
                    title="Default Configurations",
                    id="drawer",
                    padding="md",
                    lockScroll=True,
                ),
                # html.Div(id='footer',style={'background-color':'gray', 'position':'fixed', 'bottom':'0px','left':'0px','right':'0px','height':'50px', 'margin-bottom':'0px'}),
            ]
        )
        super().__init__(app=app, taskgraphs=taskgraphs)

    def callbacks(self, app, taskgraphs):
        from foreal.core import Node

        @callback(
            Input("show-config-button", "n_clicks"),
            State("cytoscape-taskgraph", "selectedNodeData"),
            State("graphs-dropdown", "value"),
            output=[Output("drawer", "opened"), Output("drawer", "children")],
            prevent_initial_call=True,
        )
        def drawer_demo(n_clicks, selectedNodeData, taskgraph_select):
            if selectedNodeData is None:
                selected_keys = []
            else:
                selected_keys = [data["label"] for data in selectedNodeData]

            taskgraph = extract_graph(taskgraph_select)
            if not isinstance(taskgraph, list):
                taskgraph = [taskgraph]

            dsk, dsk_keys = dask.base._extract_graph_and_keys(taskgraph)
            dsk_dict = dsk.to_dict()

            if not selected_keys:
                selected_keys = dsk_dict.keys()

            output = {}
            components = []
            for k in selected_keys:
                dask_node = dsk_dict[k]
                if isinstance(dask_node, tuple) and len(dask_node) > 1:
                    # the first argument is the function which is called by dask
                    dask_node = dask_node[0]

                try:
                    config_str = str(dask_node.__self__.config)
                except:
                    config_str = "No config available"

                components += [html.H3(str(k))]
                components += [html.Div(config_str)]
                # output[k] = config_str

            return True, html.Div(components)

        def get_cytoscape_elements(graph):
            if not isinstance(graph, list):
                graph = [graph]
            dsk, dsk_keys = dask.base._extract_graph_and_keys(graph)
            work = list(set(flatten(dsk_keys)))
            if not isinstance(dsk, dict):
                dsk_dict = dsk.to_dict()
            else:
                dsk_dict = dsk
            # dsk_dict = dsk
            nodes = []
            edges = []

            roots = ["#" + k for k in work]
            while work:
                new_work = {}
                for k in work:
                    nodes.append({"data": {"id": k, "label": k}})

                    current_deps = get_dependencies(dsk_dict, k, as_list=True)

                    for dep in current_deps:
                        edges.append({"data": {"source": dep, "target": k}})

                        if dep not in work:
                            new_work[dep] = True

                work = new_work

            elements = nodes + edges

            return elements

        # @callback(
        #     Output('textarea-state-example', 'value'),
        #     Input('configs-dropdown', 'value'),
        #     suppress_callback_exceptions=True)
        # def display_selected_data(selectedData):
        #     return str(catalogs[int(selectedData)])
        #     return json.dumps(selectedData, indent=2)

        @callback(
            Output("cytoscape-taskgraph", "elements"), Input("graphs-dropdown", "value")
        )
        def update_output(taskgraph_select):
            taskgraph = extract_graph(taskgraph_select)
            return get_cytoscape_elements(taskgraph)

        def extract_graph(taskgraph_select):
            app_select = taskgraph_select.split(".")[0]
            request_target = "".join(taskgraph_select.split(".")[1:])
            d = taskgraphs[app_select]
            if request_target not in d:
                return f"Invalid target for the request `{taskgraph_select}`", []
            return d[request_target]

        def concat_coords(da):
            string = ""
            for coord in da.coords:
                if len(da.coords[coord]) == 1:
                    string += " " + str(da.coords[coord].values.squeeze())
            if not string:
                string = "name"
            return string

        def extract_keys(vals):
            keys = []
            for v in vals:
                keys += list(v.__dask_keys__())

            return keys

        from dash.exceptions import PreventUpdate

        # @long_callback(
        @callback(
            Input("compute_button_id", "n_clicks"),
            State("textarea-state-example", "value"),
            State("graphs-dropdown", "value"),
            State("cytoscape-taskgraph", "selectedNodeData"),
            output=[
                Output("loading-plot", "children"),
                Output("config-computed", "children"),
                Output("cytoscape-computed", "elements"),
            ],
            # running=[
            #     (Output("compute_button_id", "disabled"), True, False),
            #     (Output("cancel_button_id", "disabled"), False, True),
            #     (
            #         Output("progress_bar", "style"),
            #         {"visibility": "visible"},
            #         {"visibility": "hidden"},
            #     ),
            # ],
            # cancel=[Input("cancel_button_id", "n_clicks")],
            # progress=[Output("progress_bar", "value"), Output("progress_bar", "max")],
            prevent_initial_call=True,
        )
        # def compute_graph(set_progress, n_clicks, value, taskgraph_select, selected_list):
        def compute_graph(n_clicks, value, taskgraph_select, selected_list):
            global app
            if selected_list is None:
                selected_keys = []
            else:
                selected_keys = [data["label"] for data in selected_list]

            ctx = dash.callback_context

            if not ctx.triggered:
                raise PreventUpdate
            else:
                input_id = ctx.triggered[0]["prop_id"].split(".")[0]

            # if '_long_callback_interval' in input_id:
            #     raise PreventUpdate

            # ctx_msg = json.dumps({
            #     'states': ctx.states,
            #     'triggered': ctx.triggered,
            #     'inputs': ctx.inputs
            # }, indent=2)

            if input_id == "compute_button_id":
                r = ast.literal_eval(value)  # convert to dict
                # request_target = 'all'

                taskgraph = extract_graph(taskgraph_select)
                if not isinstance(taskgraph, list):
                    taskgraph = [taskgraph]

                if not selected_keys:
                    selected_keys = extract_keys(taskgraph)

                # TODO: we need two types of cytoscape node select: the ones we want to
                #       configure (set them in the configuration() call). and
                #       the ones we want to output (in this case these are `selected_keys`)
                configured_collection = foreal.core.configuration(
                    taskgraph, r, optimize_graph=False
                )

                configured_graph, ck = dask.base._extract_graph_and_keys(
                    configured_collection
                )

                # get all configuration parameters formated as markdown code
                config_markdown = "# Configuration used for computing the output\n"
                for k in configured_graph:
                    # each configured foreal node is a tuple
                    # (method, input-data, input-request) if a request is available
                    # or (method, input-data) if no request is available
                    # we'd like to extract the request
                    if len(configured_graph[k]) > 2:
                        configured_request = configured_graph[k][2]
                        current_config = configured_request.get("self", {})
                    else:
                        current_config = {}
                    config_markdown += f"### {k}\n"
                    if current_config:
                        config_markdown += f"```\n{str(current_config)}\n```\n"
                    else:
                        config_markdown += f"no config available\n"

                configured_graph_keys = list(configured_graph.keys())
                new_keys = []
                for k in configured_graph_keys:
                    for sk in selected_keys:
                        if base_name(sk) == base_name(k):
                            new_keys += [k]
                xc = Delayed(new_keys, configured_graph)
                x = dask.compute(xc)

                components = []
                for example in x[0]:
                    print("current output", example)

                    if isinstance(example, xr.DataArray):
                        example.name = concat_coords(example)

                        if "frequency" in example.dims and "time" in example.dims:
                            example.coords["time"] = (
                                example.coords["time"]
                            ).values.astype(float)
                            # make sure we have enough dimensions such that holoview does not fail
                        example = example.expand_dims({"plot": [0]})

                        hv_ds = hv.Dataset(example)

                        if "frequency" in example.dims and "time" in example.dims:
                            plot = hv_ds.to(
                                hv.Image, kdims=["time", "frequency"], dynamic=False
                            ).opts(colorbar=True, xaxis=None, yaxis=None, cmap="bgy")
                        else:
                            kdim = "time"
                            if "frequency" in example.dims:
                                kdim = "frequency"

                            plot = hv_ds.to(hv.Curve, kdims=[kdim])

                        components += to_dash(shared.app, plot)[0]
                    elif isinstance(example, pd.DataFrame):
                        fig = go.Figure()
                        if (
                            "indexers.time.start" in example
                            and "indexers.time.stop" in example
                        ):
                            segments = example[example["indexers.time.start"].notna()]
                            segments = segments[segments["indexers.time.stop"].notna()]
                            for i, row in segments.iterrows():
                                start_time = pd.to_datetime(row["indexers.time.start"])
                                end_time = pd.to_datetime(row["indexers.time.stop"])
                                row_json = to_formatted_json(row)
                                y = 0
                                col = "black"
                                trace = go.Scatter(
                                    x=[
                                        start_time,
                                        end_time,
                                        end_time,
                                        start_time,
                                    ],
                                    y=[y, y, y + 1, y + 1],
                                    fill="toself",
                                    fillcolor=col,
                                    # marker={'size':12},
                                    mode="lines",
                                    hoveron="points+fills",  # select where hover is active
                                    line_color=col,
                                    showlegend=False,
                                    # line_width=0,
                                    opacity=0.5,
                                    text=str(row_json),
                                    hoverinfo="text+x",
                                )
                                fig.add_trace(trace)

                        if "indexers.time" in example:
                            events = example[example["indexers.time"].notna()]
                            for i, row in events.iterrows():
                                for time in list(row["indexers.time"]):
                                    time = pd.to_datetime(time)
                                    fig.add_vline(
                                        x=time,
                                        line_width=3,
                                        line_dash="dash",
                                        line_color="green",
                                    )
                        fig.update_xaxes(type="date")
                        fig.update_layout(
                            clickmode="event+select", modebar_add=["boxselect"]
                        )
                        time_range = r.get("indexers", {}).get("time", {})
                        if time_range:
                            start_time = pd.to_datetime(time_range["start"])
                            stop_time = pd.to_datetime(time_range["stop"])
                            fig.update_layout(xaxis_range=[start_time, stop_time])

                        components += [dcc.Graph(id="graph-dataframe", figure=fig)]
                    else:
                        components += [
                            dcc.Textarea(
                                id="text-area-plot",
                                value=str(example),
                                style={"width": "100%", "height": 200},
                            )
                        ]

                elements = get_cytoscape_elements(configured_collection)

                print("transmitting to client")
                return html.Div(children=components), config_markdown, elements
                return components
            raise PreventUpdate
            return html.Div(), []
