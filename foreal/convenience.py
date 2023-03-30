"""
Copyright 2022 Swiss Federal Institute of Technology (ETH Zurich), Matthias Meyer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""


import codecs
import io
import json
import warnings
from copy import deepcopy
from pathlib import Path

import dask
import numpy as np
import pandas as pd
from compress_pickle import dump, load
from dask.core import flatten, get_dependencies
from flask import g

# # readonly class with custom get_attr, which returns a read-only class of it if its a dict
from frozendict import frozendict

import foreal
from foreal.config import get_setting, set_setting, setting_exists

KEY_SEP = "+"


def base_name(name):
    return name.split(KEY_SEP)[0]


class NestedFrozenDict(frozendict):
    def __getitem__(self, key):
        item = super().__getitem__(key)
        if isinstance(item, dict):
            item = NestedFrozenDict(item)
        return item


class use_delayed:
    def __init__(self):
        self.prev = False
        pass

    def __enter__(self):
        if setting_exists("use_delayed"):
            self.prev = get_setting("use_delayed")
        set_setting("use_delayed", True)

    def __exit__(self, type, value, traceback):
        set_setting("use_delayed", self.prev)


class use_probe:
    def __init__(self, request):
        self.prev = None
        self.request = request
        pass

    def __enter__(self):
        if setting_exists("probe_request"):
            self.prev = get_setting("probe_request")
        set_setting("probe_request", self.request)

    def __exit__(self, type, value, traceback):
        set_setting("probe_request", self.prev)


def probe(
    graph,
    request=None,
    print_result=True,
    return_only_result=True,
    optimize_graph=False,
):
    if request is None:
        request = get_setting("probe_request")
        if request is None:
            raise RuntimeError(
                "foreal.probe requires a request or use it within a `with foreal.use_probe({...}):` statement"
            )

    configured_graph = foreal.core.configuration(
        graph, request, optimize_graph=optimize_graph
    )
    result = dask.compute(configured_graph)[0]

    if print_result:
        print(result)

    if return_only_result:
        return result
    else:
        return result, configured_graph


def indexers_to_slices(indexers):
    new_indexers = {}
    for key in indexers:
        if isinstance(indexers[key], (dict, NestedFrozenDict)):
            ni = {"start": None, "stop": None, "step": None}
            ni.update(indexers[key])
            new_indexers[key] = slice(ni["start"], ni["stop"], ni["step"])
        else:
            new_indexers[key] = indexers[key]

    return new_indexers


def exclusive_indexing(x, indexers):
    # Fake `exlusive indexing`
    drop_indexers = {
        k: indexers[k]["stop"] for k in indexers if isinstance(indexers[k], dict)
    }
    try:
        x = x.drop_sel(drop_indexers, errors="ignore")
    except Exception as ex:
        pass

    return x


def is_in_store(store, filename):
    if isinstance(store, str) or isinstance(store, Path):
        store = foreal.DirectoryStore(str(store))
    return filename in store


def requests_to_store(store, filename, requests):
    if isinstance(store, str) or isinstance(store, Path):
        store = foreal.DirectoryStore(str(store))
    StreamWriter = codecs.getwriter("utf-8")
    bytes_buffer = io.BytesIO()
    string_buffer = StreamWriter(bytes_buffer)
    # there is a bug in pandas to_json that always adds a timezone to the string
    # we bypass it by converting all timestamps to strings first
    requests = make_json_parsable(np.array(requests).tolist())
    pd.DataFrame.from_records(requests).to_json(
        string_buffer, lines=True, orient="records", date_format="iso"
    )
    store[filename] = string_buffer.getvalue()


def requests_from_store(store, filename):
    if isinstance(store, str) or isinstance(store, Path):
        store = foreal.DirectoryStore(str(store))

    bytes_buffer = io.BytesIO(store[str(filename)])
    StreamReader = codecs.getreader("utf-8")
    string_buffer = StreamReader(bytes_buffer)
    df = pd.read_json(string_buffer, lines=True)
    df = df.dropna(axis=1, how="all")
    requests = [
        {k: v for k, v in m.items() if isinstance(v, list) or pd.notnull(v)}
        for m in df.to_dict(orient="records")
    ]
    return np.array(requests)


def to_csv_with_store(store, filename, dataframe, pandas_kwargs=None):
    if pandas_kwargs is None:
        pandas_kwargs = dict()

    StreamWriter = codecs.getwriter("utf-8")
    bytes_buffer = io.BytesIO()
    string_buffer = StreamWriter(bytes_buffer)
    dataframe.to_csv(string_buffer, **pandas_kwargs)
    store[filename] = bytes_buffer.getvalue()


def read_csv_with_store(store, filename, pandas_kwargs=None):
    if pandas_kwargs is None:
        pandas_kwargs = dict()

    bytes_buffer = io.BytesIO(store[str(filename)])
    StreamReader = codecs.getreader("utf-8")
    string_buffer = StreamReader(bytes_buffer)
    return pd.read_csv(string_buffer)


def to_pickle_with_store(store, filename, object, compression="gzip"):
    bytes_buffer = io.BytesIO()
    dump(object, bytes_buffer, compression=compression)
    store[filename] = bytes_buffer.getvalue()


def read_pickle_with_store(store, filename, compression="gzip"):
    bytes_buffer = io.BytesIO(store[str(filename)])
    return load(bytes_buffer, compression=compression)


def to_datetime(x, **kwargs):
    # overwrites default
    utc = kwargs.pop("utc", True)
    if not utc:
        warnings.warn(
            "foreal's to_datetime overwrites your keyword utc argument and enforces `utc=True`"
        )
    return pd.to_datetime(x, utc=True, **kwargs).tz_localize(None)


def is_datetime(x):
    return pd.api.types.is_datetime64_any_dtype(x)


def to_datetime_conditional(x, condition=True, **kwargs):
    # converts x to datetime if condition is true or the object in condition is datetime or timedelta
    if not isinstance(condition, bool):
        condition = is_datetime(condition) or isinstance(condition, pd.Timedelta)

    if condition:
        return to_datetime(x, **kwargs)
    return x


def dict_update(base, update, convert_nestedfrozen=False):
    if not isinstance(base, (dict, NestedFrozenDict)) or not isinstance(
        update, (dict, NestedFrozenDict)
    ):
        raise TypeError(
            f"dict_update requires two dicts as input. But we received {type(base)} and {type(update)}"
        )

    for key in update:
        if isinstance(base.get(key), dict) and isinstance(update[key], dict):
            if convert_nestedfrozen:
                base[key] = dict(base[key])
            base[key] = dict_update(
                base[key], update[key], convert_nestedfrozen=convert_nestedfrozen
            )
        else:
            base[key] = update[key]

    return base


def compute(graph, request):
    """Configures the given task graph `graph` using
    foreal.core.configuration and runs the task graph
    using dask.compute

    Args:
        graph (foreal task graph): the graph which should be computed
        request (dict): request for configuring the computation

    Returns:
        any: The result of the processed task graph
    """
    configured_graph = foreal.core.configuration(graph, request)
    computed_result = dask.compute(configured_graph)
    return computed_result[0]


def make_json_parsable(requests):
    def default(o):
        if hasattr(o, "isoformat"):
            return o.isoformat()
        else:
            return str(o)

    return json.loads(json.dumps(requests, default=default))


def extract_node_from_graph(taskgraph, node_dask_key_name):
    # if not dask.is_dask_collection(taskgraph):
    if not isinstance(taskgraph, list):
        taskgraph = [taskgraph]

    dsk, dsk_keys = dask.base._extract_graph_and_keys(taskgraph)
    work = list(set(flatten(dsk_keys)))
    dsk_dict = dsk.to_dict()

    while work:
        new_work = {}
        for k in work:
            dask_node = dsk_dict[k]  # this is the function which is going to be called

            # let's see if it is a foreal node...

            # first check if it's a tuple (it should be)
            if isinstance(dask_node, tuple) and len(dask_node) > 1:
                # the first argument is the function which is called by dask
                dask_node = dask_node[0]
            # check if its a class function
            if hasattr(dask_node, "__self__"):
                # -> it's a class function
                # check if it's a foreal node
                if isinstance(dask_node.__self__, foreal.core.graph.Node):
                    # -> it's foreal node
                    # check if it's dask key name is
                    if dask_node.__self__.dask_key_name == node_dask_key_name:
                        return dask_node.__self__

            current_deps = get_dependencies(dsk_dict, k, as_list=True)

            for dep in current_deps:
                if dep not in work:
                    new_work[dep] = True

        work = new_work

    return None


from dask.delayed import Delayed

# from dask.base import is_dask_collection
# from dask.delayed import unpack_collections
# from dask.highlevelgraph import HighLevelGraph


def extract_subgraphs(taskgraph, keys, match_base_name=False):
    if not isinstance(taskgraph, list):
        taskgraph = [taskgraph]

    extracted_graph, ck = dask.base._extract_graph_and_keys(taskgraph)
    if match_base_name:
        configured_graph_keys = list(extracted_graph.keys())
        new_keys = []
        for k in configured_graph_keys:
            for sk in keys:
                if base_name(sk) == base_name(k):
                    new_keys += [k]
        keys = new_keys
    return Delayed(keys, extracted_graph)


def get_cytoscape_elements(graph):
    if not isinstance(graph, list):
        graph = [graph]
    dsk, dsk_keys = dask.base._extract_graph_and_keys(graph)
    work = list(set(flatten(dsk_keys)))
    dsk_dict = dict(dsk)
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

    return elements, roots
