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

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from benedict import benedict
from numba import jit

from foreal.convenience import read_csv_with_store
from foreal.core import Node


def set_for_keys(my_dict, key_arr, val):
    """
    Set val at path in my_dict defined by the string (or serializable object) array key_arr
    """
    current = my_dict
    for i in range(len(key_arr)):
        key = key_arr[i]
        if key not in current:
            if i == len(key_arr) - 1:
                # Found leaf (lowest-layer dictionary)
                current[key] = val
            else:
                # Generate new empty dictionary
                current[key] = {}
        else:
            if type(current[key]) is not dict:
                print("Given dictionary is not compatible with key structure requested")
                raise ValueError("Dictionary key already occupied")
            elif i == len(key_arr) - 1:
                # Update value
                # current[key] = val  # FIXME: Verify that this is the intended behaviour
                pass

        # Enter nested dictionary
        current = current[key]

    return my_dict


def to_formatted_json(df, sep="."):
    def process_row(row):
        parsed_row = {}
        for idx, val in row.iteritems():
            if isinstance(val, (np.ndarray, list)) or pd.notnull(val):
                keys = idx.split(sep)
                parsed_row = set_for_keys(parsed_row, keys, val)
        return parsed_row

    result = []
    if isinstance(df, pd.Series):
        parsed_row = process_row(df)
        result.append(parsed_row)
    else:
        for _, row in df.iterrows():
            parsed_row = process_row(row)
            result.append(parsed_row)
    return result


@jit(nopython=True)
def check_overlap(data0, data1, additional_condition=None, closed="neither"):
    """Checks the overlap of two sorted numpy arrays

    Arguments:
        data0 {numpy array} -- First numpy array
        data1 {numpy array} -- Second list of xarray indexers. Must be sorted primarily by first column and secondarily by second column

    Keyword Arguments:
        additional_condition {list} -- List of additional dimensions which should be checked for overlap (default: {[]})
        closed {string} -- {'both’, 'neither’}, default 'neither’
    Returns:
        list -- List of indices of which items overlap. [...,[i,j],...] where item i of data0 overlaps with item j of data1
    """

    # x = [
    #             [
    #                 s["indexers"]["time"]["start"],
    #                 s["indexers"]["time"]["stop"],
    #             ]
    #             for s in requests
    #         ]
    # df = pd.DataFrame(x,columns=['start','stop'])

    # Ensure that data1 is of larger size
    flipped = len(data0) > len(data1)
    if flipped:
        tmp_data1 = data1
        data1 = data0
        data0 = tmp_data1

    overlap_indices = []
    start_idx = 0

    for i in range(len(data0)):
        data0_start = data0[i, 0]
        data0_end = data0[i, 1]

        for j in range(start_idx, len(data1)):
            data1_start = data1[j, 0]
            data1_end = data1[j, 1]

            # first condition: data0[i] is entirely before data1[j], the remaining data1 are not relevant for data0[i] (sorted list data1)
            if closed == "neither" or closed == "left":
                cond0 = data0_end < data1_start
            else:
                cond0 = data0_end <= data1_start

            if cond0:
                break

            # second condition: data0[i] is fully after data1[j], all items before data1[j] can be ignored for data0[i+1] (sorted list data0)
            if closed == "neither" or closed == "right":
                cond1 = data0_start > data1_end
            else:
                cond1 = data0_start >= data1_end

            if cond1:
                # This only holds if data1 is sorted by both start (primary) and end index (secondary)
                start_idx = j

            if not (cond0 or cond1):
                # overlap on index dimension
                # check other dimensions
                overlap = True
                # for dim in dims:
                #     if dim == index_dim:
                #         continue
                #     d0_start = data0[i]['indexers'][dim]['start']
                #     d0_end = data0[i]['indexers'][dim]['stop']
                #     d1_start = data1[j]['indexers'][dim]['start']
                #     d1_end = data1[j]['indexers'][dim]['stop']
                #     if (d0_end < d1_start) or (d0_start > d1_end):
                #         overlap = False
                if additional_condition is not None:
                    if not flipped:
                        overlap = additional_condition(i, j)
                    else:
                        overlap = additional_condition(j, i)

                if overlap:
                    if not flipped:
                        overlap_indices.append([int(i), int(j)])
                    else:
                        overlap_indices.append([int(j), int(i)])

    return overlap_indices


def requests_to_file(filename, requests):
    pd.DataFrame.from_records(requests).to_json(
        filename, lines=True, orient="records", date_format="iso"
    )


def requests_from_file(filename):
    filename = Path(filename)
    if not filename.exists():
        raise RuntimeError(f"File doe not exist {str(filename)}")
    df = pd.read_json(str(filename), lines=True)
    df = df.dropna(axis=1, how="all")
    requests = [
        {k: v for k, v in m.items() if isinstance(v, list) or pd.notnull(v)}
        for m in df.to_dict(orient="records")
    ]
    return np.array(requests)


# TODO: use schema with Use() https://pypi.org/project/schema/
def apply_schema(data, schema, sub_dict_key=None):
    data = benedict(data)
    if sub_dict_key is not None:
        schema = schema[".".join(sub_dict_key)]

        sub_dict_key = list(sub_dict_key)
    else:
        sub_dict_key = []
    for key in schema:
        if isinstance(schema[key], dict):
            apply_schema(data, schema, sub_dict_key=tuple(sub_dict_key + [key]))
            data[sub_dict_key + [key]] = dict(data[sub_dict_key + [key]])

        elif schema[key] == "datetimeInterval":
            bene_key = tuple(sub_dict_key + [key])
            start_key = tuple(list(bene_key) + ["start"])
            stop_key = tuple(list(bene_key) + ["stop"])
            start = pd.to_datetime(data[start_key]).tz_localize(None)
            stop = pd.to_datetime(data[stop_key]).tz_localize(None)
            data[bene_key] = pd.Interval(start, stop, closed="left")

    return dict(data)


def normalize_keys(d, subkeys=[]):
    keys = []
    for k in d:
        if not isinstance(d[k], dict):
            keys += [".".join(subkeys) + "." + k]
        else:
            sk = deepcopy(subkeys) + [k]
            keys += normalize_keys(d[k], sk)
    return keys


def mask_request(data, request, schema, mask=None, sub_dict_key=None):
    if mask is None:
        mask = np.ones((len(data),)).astype(np.bool)
    # masked = np.array(data)

    if sub_dict_key is not None:
        schema = schema[sub_dict_key]
        sub_dict_key = list(sub_dict_key)
    else:
        sub_dict_key = []

    for key in schema:
        if isinstance(schema[key], dict):
            mask = mask_request(
                data,
                request,
                schema,
                mask=mask,
                sub_dict_key=tuple(sub_dict_key + [key]),
            )
        else:
            if key in schema and "Interval" in schema[key]:
                # overlap comparison
                bene_key = tuple(sub_dict_key + [key])

                interval_list_request = [r[bene_key] for r in request]
                interval_array_request = pd.arrays.IntervalArray(interval_list_request)

                interval_array_masked = data[".".join(bene_key)]

                # overlap = interval_array_masked.overlaps(interval_list_request)
                if len(interval_array_request) <= 1:
                    overlap = pd.arrays.IntervalArray(interval_array_masked).overlaps(
                        interval_list_request[0]
                    )
                else:
                    interval_array_masked = pd.arrays.IntervalArray(
                        interval_array_masked
                    )
                    overlap = check_overlap(
                        interval_array_masked, interval_array_request
                    )

                mask_indices = np.arange(len(mask))
                mask_indices_overlap = mask_indices[mask][overlap]

                mask = np.zeros((len(data),)).astype(np.bool)
                mask[mask_indices_overlap] = True
            elif "list" in schema[key]:
                raise NotImplementedError(
                    "lists are not supported in BoundingBoxAnnotation request"
                )
            else:
                # direct comparison
                bene_key = tuple(sub_dict_key + [key])
                # masked = data[mask]
                # subset_mask = np.array([request[bene_key] == benedict(masked[i])[bene_key] if bene_key in benedict(masked[i]) else False for i in range(len(masked))]).astype(np.bool)
                masked_indices = []
                for r in request:
                    subset_mask = data[mask][".".join(bene_key)] == r[bene_key]

                    mask_indices = np.arange(len(mask))
                    mask_indices_overlap = mask_indices[mask][subset_mask]

                    mask = np.zeros((len(data),)).astype(np.bool)
                    mask[mask_indices_overlap] = True

                    masked_indices.append(mask.nonzero())
                if len(request) > 1:
                    # TODO: Do something with mask_indices
                    pass

    return mask


def to_numeric(requests, interval_key, schema):
    interval_array = pd.arrays.IntervalArray(requests[interval_key])
    # convert to datetime if necessary:
    start_values = interval_array.left
    stop_values = interval_array.right
    if schema[interval_key] == "datetimeInterval":
        start_values = pd.to_datetime(start_values)
        stop_values = pd.to_datetime(stop_values)

    start_values = pd.to_numeric(start_values).to_numpy()
    stop_values = pd.to_numeric(stop_values).to_numpy()
    return np.stack([start_values, stop_values], axis=1)


class BoundingBoxAnnotation(Node):
    def __init__(
        self,
        requests_or_filename=None,
        store=None,
        converters=None,
        schema=None,
        target_subset=None,
        only_targets=True,
        closed="neither",
        **kwargs,
    ):
        # Make sure default arguments are non-mutable
        if converters is None:
            converters = {}
        if schema is None:
            schema = {}

        super().__init__(
            store=store,
            converters=converters,
            kwargs=kwargs,
        )

        if (
            isinstance(requests_or_filename, (Path, str))
            and Path(requests_or_filename).exists()
        ):
            self.requests = requests_from_file(requests_or_filename).tolist()
        elif isinstance(requests_or_filename, (np.ndarray, list)):
            self.requests = requests_or_filename
        else:
            raise RuntimeError(f"Could not handle input: {requests_or_filename}")

        self.schema = benedict(schema)
        self.closed = closed
        #        self.requests = np.array([apply_schema(r,schema) for r in self.requests])
        sub_dict_key = ["indexers", "time"]

        # TODO: fixme! below
        for r in self.requests:
            if isinstance(r["indexers"]["time"], list):
                t = r["indexers"]["time"]
                r["indexers"]["time"] = {"start": t, "stop": t}

        self.requests = pd.json_normalize(self.requests)

        self.requests.sort_values(
            [".".join(sub_dict_key + ["start"]), ".".join(sub_dict_key + ["stop"])],
            inplace=True,
        )
        self.requests[".".join(sub_dict_key)] = pd.arrays.IntervalArray.from_arrays(
            pd.to_datetime(
                self.requests[".".join(sub_dict_key + ["start"])]
            ).dt.tz_localize(None),
            pd.to_datetime(
                self.requests[".".join(sub_dict_key + ["stop"])]
            ).dt.tz_localize(None),
        )

        interval_keys = [".".join(sub_dict_key)]
        self.data0 = {}
        for interval_key in interval_keys:
            self.data0[interval_key] = to_numeric(
                self.requests, interval_key, self.schema
            )
        # data0 = pd.arrays.IntervalArray(self.requests[interval_key])
        # data0_starts = pd.to_numeric(pd.to_datetime(data0.left))
        # data0_ends = pd.to_numeric(pd.to_datetime(data0.right))
        # self.data0 = np.stack([data0_starts, data0_ends], axis=1)
        return

    def __dask_tokenize__(self):
        return (BoundingBoxAnnotation,)

    def configure(self, requests=None):
        """Default configure for BoundingBoxAnnotation nodes
        Arguments:
            requests {list} -- List of requests

        Returns:
            dict -- Original request or merged requests
        """
        request = super().configure(requests)  # merging request here

        new_request = {"requires_request": True}
        new_request.update(request)

        return new_request

    def forward(self, data, request):
        # requested_segment = indexers_to_overlap_array(request["indexers"])

        if isinstance(request, dict) and self.dask_key_name in request:
            request = request[self.dask_key_name]["requests"]

        singleton = not isinstance(request, (np.ndarray, list))
        if singleton:
            request = [request]

        # test_request = {"indexers":{"time":{"start":"2017-05-27T04:02:28.619Z","stop":"2017-05-27T04:02:58.619Z"}},"station":"ILL04","network":"XP","location":""}
        # Check for types: Either type interval or non-interval
        # Go through all non-interval types
        schema = benedict(self.schema)
        # request = benedict(request)
        # request = [apply_schema(benedict(r),schema) for r in request]

        # TODO: fixme! below
        for r in request:
            if isinstance(r["indexers"]["time"], list):
                t = r["indexers"]["time"]
                r["indexers"]["time"] = {"start": t, "stop": t}

        # TODO:
        # Define a function for overlap computation and one for equality checking
        # Input to each function is an array containing indices for data0 (internal data) and data1 (request)
        # and both data0 and data1
        # The
        request = pd.json_normalize(request)

        sub_dict_key = ["indexers", "time"]
        # TODO: find all interval keys in schema
        interval_keys = [".".join(sub_dict_key)]

        ######## INTERVAL OVERLAP DETECTION ##########
        self_requests_subset = None
        request_subset = None
        for interval_key in interval_keys:
            request[interval_key] = pd.arrays.IntervalArray.from_arrays(
                pd.to_datetime(
                    request[".".join(sub_dict_key + ["start"])]
                ).dt.tz_localize(None),
                pd.to_datetime(
                    request[".".join(sub_dict_key + ["stop"])]
                ).dt.tz_localize(None),
            )

            if self_requests_subset is None:
                data0 = self.data0[interval_key]  # use precomputed
                data1 = to_numeric(request, interval_key, schema)
                self_requests_subset = self.requests
                request_subset = request
            else:
                data0 = to_numeric(self_requests_subset, interval_key, schema)
                data1 = to_numeric(request_subset, interval_key, schema)

            # data0 = self.data0[interval_key] # use precomputed
            # data1 = to_numeric(request,interval_key,schema)

            # data1 = pd.arrays.IntervalArray(request[interval_key])
            # data1_starts = pd.to_numeric(pd.to_datetime(data1.left)).to_numpy()
            # data1_ends = pd.to_numeric(pd.to_datetime(data1.right)).to_numpy()
            # data1 = np.stack([data1_starts, data1_ends], axis=1)

            # sort start_time primary, end_time secondary
            # (lexsort treats the last element in given sequence as primary key)
            data0_ind = np.lexsort((data0[:, 1], data0[:, 0]))
            data1_ind = np.lexsort((data1[:, 1], data1[:, 0]))

            overlap = np.array(
                check_overlap(data0[data0_ind], data1[data1_ind], closed=self.closed)
            )

            # Check whether no overlap has been detected
            if len(overlap) == 0:
                if singleton:
                    return {}
                else:
                    return [{}]

            # Reverse the indices to the original sorting
            overlap[:, 0] = data0_ind[overlap[:, 0]]
            overlap[:, 1] = data1_ind[overlap[:, 1]]

            self_requests_subset = self_requests_subset.iloc[overlap[:, 0]]
            request_subset = request_subset.iloc[overlap[:, 1]]

            # sub0 = self.requests.iloc[overlap[:, 0]]
            # sub1 = request.iloc[overlap[:, 1]]

        ######## EQUALITY DETECTION ##########
        mask = None
        ms = deepcopy(schema)
        # remove all interval keys because we've dealt with that
        # before during overlap detection
        for ik in interval_keys:
            ms.pop(ik, None)

        norm_schema = normalize_keys(ms)
        for key in norm_schema:
            m = self_requests_subset[key].eq(request_subset[key].values)
            if mask is None:
                mask = m
            else:
                mask = np.logical_and(mask, m)

        subo = overlap[mask.values]

        out = [[] for _ in range(len(request))]
        for i in range(len(subo)):
            o = subo[i]
            j, k = o[0], o[1]
            out_k = self.requests.iloc[j : j + 1]
            out_k = to_formatted_json(out_k, sep=".")[0]
            out[k].append(out_k)

        if singleton:
            out = out[0]

        return out

        # FIXME: Mask will never be applied
        mask = mask_request(self.requests, request, schema)
        return deepcopy(self.requests[mask])
