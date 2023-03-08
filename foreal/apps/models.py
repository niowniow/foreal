from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import classification_report

import foreal
from foreal.convenience import dict_update, indexers_to_slices
from foreal.core import Node
from foreal.core.graph import Node, NodeFailedException


def get_segments(
    dataset_scope,
    dims,
    stride=None,
    ref=None,
    mode="fit",
    minimal_number_of_segments=0,
    timestamps_as_strings=False,
    utc_no_tz=True,
):
    # modified from and thanks to xbatcher: https://github.com/rabernat/xbatcher/
    if isinstance(mode, str):
        mode = {dim: mode for dim in dims}

    if stride is None:
        stride = {}

    if ref is None:
        ref = {}

    dim_slices = []
    for dim in dims:
        # if dataset_scope is None:
        #     segment_start = 0
        #     segment_end = ds.sizes[dim]
        # else:
        size = dims[dim]
        _stride = stride.get(dim, size)

        if isinstance(dataset_scope[dim], list):
            segment_start = 0
            segment_end = len(dataset_scope[dim])
        else:
            segment_start = foreal.to_datetime_conditional(
                dataset_scope[dim]["start"], dims[dim]
            )
            segment_end = foreal.to_datetime_conditional(
                dataset_scope[dim]["stop"], dims[dim]
            )

            if mode[dim] == "overlap":
                # TODO: add options for closed and open intervals
                # first get the lowest that window that still overlaps with our segment
                segment_start = segment_start - np.floor(size / _stride) * _stride
                # then align to the grid if necessary
                if dim in ref:
                    segment_start = (
                        np.ceil((segment_start - ref[dim]) / _stride) * _stride
                        + ref[dim]
                    )
                segment_end = segment_end + size
            elif mode[dim] == "fit":
                if dim in ref:
                    segment_start = (
                        np.floor((segment_start - ref[dim]) / _stride) * _stride
                        + ref[dim]
                    )
            else:
                RuntimeError(f"Unknown mode {mode[dim]}. It must be `fit` or `overlap`")

        if isinstance(
            dims[dim], pd.Timedelta
        ):  # or isinstance(dims[dim], dt.timedelta):
            # TODO: change when xarray #3291 is fixed
            iterator = pd.date_range(segment_start, segment_end, freq=_stride)
            segment_end = pd.to_datetime(segment_end)
        else:
            iterator = range(segment_start, segment_end, _stride)

        slices = []
        for start in iterator:
            end = start + size
            if end <= segment_end or (
                len(slices) < minimal_number_of_segments
                and not isinstance(dataset_scope[dim], list)
            ):
                if foreal.is_datetime(start):
                    if utc_no_tz:
                        start = pd.to_datetime(start, utc=True).tz_localize(None)
                    if timestamps_as_strings:
                        start = start.isoformat()
                if foreal.is_datetime(end):
                    if utc_no_tz:
                        end = pd.to_datetime(end, utc=True).tz_localize(None)
                    if timestamps_as_strings:
                        end = end.isoformat()

                if isinstance(dataset_scope[dim], list):
                    slices.append(dataset_scope[dim][start:end])
                else:
                    slices.append({"start": start, "stop": end})

        dim_slices.append(slices)

    import itertools

    all_slices = []
    for slices in itertools.product(*dim_slices):
        selector = {key: slice for key, slice in zip(dims, slices)}
        all_slices.append(selector)

    return np.array(all_slices)


class ModelWrapper(Node):
    def __init__(self, model, convert_to_targets=None, add_logit=True):
        super().__init__()
        self.model = model
        self.convert_to_targets = convert_to_targets
        self.add_logit = add_logit

    def __dask_tokenize__(self):
        return (ModelWrapper,)

    def configure(self, requests=None):
        request = super().configure(requests)

        new_request = {"requires_request": True}
        new_request["clone_dependencies"] = request["self"]["batch_requests"]
        dict_update(
            new_request, {"self": {"batch_requests": request["self"]["batch_requests"]}}
        )
        return new_request

    def forward(self, data, request):
        if isinstance(data, NodeFailedException):
            # This will probably not happen
            return NodeFailedException("Failed to load any data during model inference")

        # data = data[0]

        # collate data and take care that none of it is faulty
        collated_data = []
        collated_requests = []
        for i in range(len(data)):
            if isinstance(data[i], NodeFailedException):
                continue
            collated_data.append(data[i])
            collated_requests.append(request["self"]["batch_requests"][i])

        if not collated_data:
            return NodeFailedException("No data available to collate")

        if isinstance(collated_data[0], xr.DataArray):
            collated_data = xr.concat(collated_data, "collated")
        elif isinstance(collated_data[0], np.ndarray):
            collated_data = np.stack(collated_data)
        else:
            raise RuntimeError("Data type not support got", type(collated_data[0]))
        predictions = self.model.predict(collated_data)
        if self.convert_to_targets is not None:
            return self.convert_to_targets(predictions, collated_requests)

        return (predictions, collated_requests)


class ModelWrapperGenerator(Node):
    def __init__(
        self,
        model,
        batch_size=1,
        dim="time",
        classification_scope=None,
        segment_slice=None,
        segment_stride=None,
        mode="fit",
        ref=None,
        convert_to_targets=None,
        add_logit=True,
        device="cpu",
    ):
        if callable(classification_scope):
            self.classification_scope = classification_scope
            classification_scope = None
        else:
            self.classification_scope = None

        if callable(segment_slice):
            self.segment_slice = segment_slice
            segment_slice = None
        else:
            self.segment_slice = None

        if callable(segment_stride):
            self.segment_stride = segment_stride
            segment_stride = None
        else:
            self.segment_stride = None

        super().__init__(
            batch_size=batch_size,
            dim=dim,
            classification_scope=classification_scope,
            segment_slice=segment_slice,
            segment_stride=segment_stride,
            mode=mode,
            ref=ref,
        )
        self.model = model
        # self.model = self.model.to(device)
        self.device = device

        # self.segment_slice = {"time": foreal.to_timedelta(30, "seconds")}
        # self.segment_stride = {"time": foreal.to_timedelta(30, "seconds")}
        self.convert_to_targets = convert_to_targets
        self.add_logit = add_logit

    def __dask_tokenize__(self):
        return (ModelWrapperGenerator,)

    def configure(self, requests=None):
        request = super().configure(requests)
        rs = request["self"]
        new_request = {"requires_request": True}

        if rs["dim"] == "index":
            indexers = indexers_to_slices(rs["indexers"])["index"]
            if isinstance(indexers, slice):
                if indexers.start is None or indexers.stop is None:
                    raise RuntimeError(
                        "indexer with dim index must have a start and stop value"
                    )
                indexers = list(
                    range(indexers.start, indexers.stop, indexers.step or 1)
                )
            if not isinstance(indexers, list):
                raise RuntimeError(
                    "indexer with dim index must be of type list, dict or slice"
                )
            segments = [{"index": {"start": x, "stop": x + 1}} for x in indexers]
        else:

            def get_value(attr_name):
                # decide if we use the attribute provided in the request or
                # from a callback provided at initialization

                if rs.get(attr_name, None) is None:
                    # there is no attribute in the request, check for callback
                    callback = getattr(self, attr_name)
                    if callback is not None and callable(callback):
                        value = callback(request)
                    else:
                        RuntimeError("No valid classification_scope provided")
                else:
                    value = rs[attr_name]
                return value

            classification_scope = get_value("classification_scope")
            segment_slice = get_value("segment_slice")
            segment_stride = get_value("segment_stride")
            segments = get_segments(
                classification_scope,
                segment_slice,
                segment_stride,
                ref=rs["ref"],
                mode=rs["mode"],
                timestamps_as_strings=True,
                minimal_number_of_segments=1,
            )

        batch_size = rs["batch_size"]
        num_batches = int(np.ceil(len(segments) / batch_size))

        # count number of model instances required based on
        # - batch size
        # - number of slices
        # and create seperate request for them
        cloned_requests = []
        cloned_models = []
        for batch_num in range(num_batches):
            batch_requests = []
            # batch_slices = slices[i:i+batch_size]
            for idx in range(batch_size):
                i = batch_num * batch_size + idx
                batch_request = deepcopy(request)

                current_slice = segments[i]
                dict_update(batch_request, {"indexers": current_slice})
                batch_requests += [batch_request]

            sub_request = {"config": {"batch_requests": batch_requests}}
            cloned_requests += [sub_request]
            cloned_model = ModelWrapper(
                self.model,
                convert_to_targets=self.convert_to_targets,
            )
            cloned_model.dask_key_name = self.dask_key_name + "_model"
            cloned_models += [cloned_model.forward]

        # Insert predecessor

        new_request["clone_dependencies"] = cloned_requests
        new_request["insert_predecessor"] = cloned_models

        return new_request

    def forward(self, data, request):
        return data
