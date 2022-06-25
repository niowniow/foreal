import numpy as np
import pandas as pd


def get_dataset_scopes(dims, dataset_scope, stride=None):
    # Based on code from xbatcher https://github.com/rabernat/xbatcher/
    """The MIT License (MIT)
    Copyright (c) 2016 Ryan Abernathey"""

    # Make sure default arguments are non-mutable
    if stride is None:
        stride = {}

    dim_slices = []
    for dim in dims:
        # if dataset_scope is None:
        #     segment_start = 0
        #     segment_end = ds.sizes[dim]
        # else:
        segment_start = dataset_scope[dim]["start"]
        segment_end = dataset_scope[dim]["stop"]

        size = dims[dim]
        _stride = stride.get(dim, size)

        if isinstance(
            dims[dim], pd.Timedelta
        ):  # or isinstance(dims[dim], dt.timedelta):
            # TODO: change when xarray #3291 is fixed
            iterator = pd.date_range(
                segment_start, segment_end, freq=_stride
            ).tz_localize(None)
            segment_end = pd.to_datetime(segment_end).tz_localize(None)
        else:
            iterator = range(segment_start, segment_end, _stride)

        slices = []
        for start in iterator:
            end = start + size
            if end <= segment_end:
                slices.append({"start": start.isoformat(), "stop": end.isoformat()})

        dim_slices.append(slices)

    import itertools

    all_slices = []
    for slices in itertools.product(*dim_slices):
        selector = {key: slice for key, slice in zip(dims, slices)}
        all_slices.append(selector)

    return np.array(all_slices)
