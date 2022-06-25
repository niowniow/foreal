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


import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr
from xarray.core import dataarray

from foreal.convenience import exclusive_indexing, indexers_to_slices
from foreal.core import Node


class MinMaxDownsampling(Node):
    def __init__(
        self, rate=1, dim=None, input_sampling_rate=None, reference_value=None
    ):
        # dim is by default the last dimension
        # since we always choose two values (min and max) per bucket the
        # the internal downsampling rate must be of factor two larger than
        # the effective (and desired) downsampling rate
        if reference_value is None:
            reference_value = "2000-01-01"
        self.reference_value = pd.to_datetime(reference_value)
        super().__init__(rate=rate, dim=dim, input_sampling_rate=input_sampling_rate)
        # self.rate = rate * 2
        # self.dim = dim

    def align_indexers(
        self, index, stride, input_sampling_rate, round_type="floor", end_index=None
    ):
        region = stride / input_sampling_rate  # seconds
        region = pd.to_timedelta(region, "seconds")
        # FIXME: make this index type dependant

        index = pd.to_datetime(index)

        chunk_index = (
            np.floor((index - self.reference_value) / region) * region
            + self.reference_value
        )

        # if end_index is not None:
        #    return

        if round_type == "ceil":
            chunk_index += region

        return chunk_index

    def configure(self, requests):
        """

        Arguments:
            request {list} -- List of requests

        Returns:
            dict -- Original, updated or merged request(s)
        """
        requests = super().configure(requests)

        #        print('st',requests)
        requests = deepcopy(dict(requests))

        indexers = requests["indexers"]

        if self.config["dim"] != "time":
            raise NotImplementedError()

        window_length_seconds = pd.to_timedelta(
            self.config["rate"] / self.config["input_sampling_rate"], "seconds"
        )
        input_sampling_rate = self.config["input_sampling_rate"]
        rate = self.config["rate"]

        requests["self"] = {"indexers": deepcopy(requests["indexers"])}
        requests["indexers"][self.config["dim"]]["start"] = (
            pd.to_datetime(requests["indexers"][self.config["dim"]]["start"])
            - window_length_seconds
        )
        #        requests['indexers'][self.config['dim']]['stop'] = pd.to_datetime(requests['indexers'][self.config['dim']]['stop']) + window_length_seconds

        if self.config["dim"] in indexers:
            # align the start of the indexing to make the spectogram windows evenly distributed
            requests["indexers"][self.config["dim"]]["start"] = self.align_indexers(
                requests["indexers"][self.config["dim"]]["start"],
                rate,
                input_sampling_rate,
            )
            requests["indexers"][self.config["dim"]]["stop"] = self.align_indexers(
                requests["indexers"][self.config["dim"]]["stop"],
                rate,
                input_sampling_rate,
                round_type="ceil",
            )

        #        print('en',requests)

        requests["requires_request"] = True
        return requests

    def forward(self, data=None, request=None):
        dim = request["dim"]
        rate = request["rate"] * 2
        if dim is None:
            dim = data.dims[-1]

        indexers = None
        if "self" in request:
            indexers = request["self"]["indexers"]
        elif "indexers" in request:
            indexers = request["indexers"]

        # # FIXME: START: regular downsampling for testing purposes
        # x_ds = data.isel({dim: slice(None, None, request["rate"])})
        # x_ds = x_ds.dropna(dim=dim, how="all")

        # if indexers is not None:
        #     i2slices = indexers_to_slices(indexers)
        #     x_ds = x_ds.sel(i2slices)
        #     drop_indexers = {k: indexers[k]["stop"] for k in indexers}
        #     try:
        #         x_ds = x_ds.drop_sel(drop_indexers, errors="ignore")
        #     except Exception as ex:
        #         pass

        # return x_ds
        # # FIXME: END: regular downsampling for testing purposes

        # rolling = data.rolling(dim={dim: rate})
        # x_min = rolling.construct("buckets", stride=rate)
        # print(x_min)
        # x_min = x_min.min("buckets")

        # # rolling = data.rolling(dim={dim: rate},center=True)
        # x_max = rolling.construct("buckets", stride=rate)
        # print(x_max)
        # x_max = x_max.max("buckets")

        rolling = data.rolling(dim={dim: rate}, min_periods=rate, stride=rate)
        x_min = rolling.min().dropna(dim)
        x_max = rolling.max().dropna(dim)
        # print("xmin",x_min)
        # print("xmax",x_max)
        # print(data)
        # print("xr",rolling.construct("buckets"))
        # print(request)

        if "sampling_rate" in data.attrs:
            step = 1 / (data.attrs["sampling_rate"] / (rate))
        elif (
            request is not None and request.get("input_sampling_rate", None) is not None
        ):
            step = 1 / (request["input_sampling_rate"] / (rate))
        elif len(x_max[dim]) > 1:
            step = x_max[dim][1] - x_max[dim][0]
            # print(data)
            # raise RuntimeError("Please provide a sampling rate")
            # TODO: Computing based on data might lead to issues if the first
            # two samples are unlike the rest and/or irregular sampling intervals
        else:
            raise RuntimeError("Could not determine a proper step size")

        stepo = step
        if pd.api.types.is_datetime64_any_dtype(data[dim]):
            step = pd.to_timedelta(step, "s")
            if isinstance(indexers[dim], dict):
                indexers[dim]["start"] = pd.to_datetime(indexers[dim]["start"])
                indexers[dim]["stop"] = pd.to_datetime(indexers[dim]["stop"])

        # print(x_max[dim])
        x_max[dim] = x_max[dim] + step / 2
        # print(x_max[dim])

        #        a = x_min[dim].values
        #        b = x_max[dim].values
        #        c = np.empty((a.size + b.size,), dtype=a.dtype)
        #        c[0::2] = a
        #        c[1::2] = b
        #        coords = data.coords
        #        coords[dim] = c
        #        template_da = xr.DataArray(template,dims=data.dims,coords=coords,name=data.name)
        #
        # x_ds = xr.concat([x_min, x_max], dim+'_concat')
        x_ds = xr.concat([x_min, x_max], dim)
        mask = np.zeros((x_ds.sizes[dim],))

        mask[1::2] = 1

        mask = mask.astype(np.bool)

        x_ds[{dim: ~mask}] = x_min.values
        x_ds[{dim: mask}] = x_max.values

        # mask = np.zeros(x_ds.shape)
        # mask[...,::2] = 1
        # x_ds = xr.where(mask.astype(np.bool),x_min,x_max)

        #        x_ds.loc[dict(time=slice(1,None,2))] = x_max.values
        new_time = np.zeros((x_ds.sizes[dim],), dtype=x_min[dim].values.dtype)
        new_time[::2] = x_min[dim].values
        new_time[1::2] = x_max[dim].values
        x_ds[dim] = new_time
        # x_ds['time'][slice(None,None,2)] = x_min['time'].values
        # x_ds['time'][slice(1,None,2)] = x_max['time'].values

        #        dim_index = x_ds.dims.index(dim)
        #        print(dim_index)
        #        mask = np.zeros(x_ds.shape)

        #        mask[...,::2] = 1
        #        print(x_ds.shape,mask.shape)
        # x_ds = x_ds.where(mask.astype(np.bool),x_min,x_max)
        #        print(x_ds)
        #        exit()
        # FIXME: too expensive
        #        x_ds = x_ds.sortby(dim)
        #        x_ds = x_ds.chunk({dim:x_ds.sizes[dim]})
        x_ds.attrs["sampling_rate"] = 1 / stepo
        if "sampling_rate" in data.attrs:
            x_ds.attrs["sampling_rate"] = data.attrs["sampling_rate"] / rate * 2

        if indexers is not None:
            # print(x_ds,indexers)
            #        x_ds = x_ds.dropna(dim=dim, how="all")

            try:
                i2slices = indexers_to_slices(indexers)
                x_ds = x_ds.sel(i2slices)
            except Exception as ex:
                not_null = pd.notnull(x_ds[dim])
                x_ds = x_ds.isel({dim: not_null})
                i2slices = indexers_to_slices(indexers)

                # print(x_ds)

                # i2slices['time'] = slice(i2slices['time'].start.isoformat(),i2slices['time'].stop.isoformat())

                # print('zoi',i2slices)
                # print(x_ds.values)
                # # x_ds = x_ds.sel(i2slices)
                # time = x_ds['time'].values
                # print(time[:10])
                # is_sorted = np.all(time[:-1] <= time[1:])
                # sorted_time = np.sort(time)
                # indexes = (time != sorted_time).nonzero()

                # print(is_sorted)
                # print(indexes)
                x_ds = x_ds.sel(i2slices)

                # assert False
            # Fake `excluding indexing`
            drop_indexers = {k: indexers[k]["stop"] for k in indexers}
            try:
                x_ds = x_ds.drop_sel(drop_indexers, errors="ignore")
            except Exception as ex:
                pass

        return x_ds
        # print(x_ds[dim])
        # exit()
        x_ds = x_ds.rename({dim: dim + "_rn"})
        x_ds = x_ds.stack({dim: (dim + "_rn", dim + "_concat")})
        print(x_ds)

        # x_ds = x_ds.drop_vars('time')
        print(x_ds)
        print(x_ds.indexes[dim])
        x_ds[dim] = x_ds.indexes[dim].droplevel(1)
        print(x_ds.indexes[dim])
        exit()
        # del x_ds[dim+'_concat']
        # x_ds = x_ds.dropna(dim)

        # x_ds = x_ds.sum(dim+'_concat')

        return x_ds


class MaxDownsampling(Node):
    def __init__(
        self, rate=1, dim=None, input_sampling_rate=None, reference_value=None
    ):
        # dim is by default the last dimension
        if reference_value is None:
            reference_value = "2000-01-01"
        self.reference_value = pd.to_datetime(reference_value)
        super().__init__(rate=rate, dim=dim, input_sampling_rate=input_sampling_rate)

    def align_indexers(
        self, index, stride, input_sampling_rate, round_type="floor", end_index=None
    ):
        region = stride / input_sampling_rate  # seconds
        region = pd.to_timedelta(region, "seconds")
        # FIXME: make this index type dependant

        index = pd.to_datetime(index)

        chunk_index = (
            np.floor((index - self.reference_value) / region) * region
            + self.reference_value
        )

        # if end_index is not None:
        #    return

        if round_type == "ceil":
            chunk_index += region

        return chunk_index

    def configure(self, requests):
        """

        Arguments:
            request {list} -- List of requests

        Returns:
            dict -- Original, updated or merged request(s)
        """
        requests = super().configure(requests)

        #        print('st',requests)
        requests = deepcopy(dict(requests))

        indexers = requests["indexers"]

        if self.config["dim"] != "time":
            raise NotImplementedError()

        input_sampling_rate = self.config["input_sampling_rate"]
        if input_sampling_rate is not None:
            window_length_seconds = pd.to_timedelta(
                self.config["rate"] / input_sampling_rate, "seconds"
            )
            rate = self.config["rate"]

            requests["self"] = {"indexers": deepcopy(requests["indexers"])}
            requests["indexers"][self.config["dim"]]["start"] = (
                pd.to_datetime(requests["indexers"][self.config["dim"]]["start"])
                - window_length_seconds
            )
            #        requests['indexers'][self.config['dim']]['stop'] = pd.to_datetime(requests['indexers'][self.config['dim']]['stop']) + window_length_seconds

            if self.config["dim"] in indexers:
                # align the start of the indexing to make the spectogram windows evenly distributed
                requests["indexers"][self.config["dim"]]["start"] = self.align_indexers(
                    requests["indexers"][self.config["dim"]]["start"],
                    rate,
                    input_sampling_rate,
                )
                requests["indexers"][self.config["dim"]]["stop"] = self.align_indexers(
                    requests["indexers"][self.config["dim"]]["stop"],
                    rate,
                    input_sampling_rate,
                    round_type="ceil",
                )

        requests["requires_request"] = True
        return requests

    def forward(self, data=None, request=None):
        dim = request["dim"]
        rate = request["rate"]
        if dim is None:
            dim = data.dims[-1]

        indexers = None
        if "self" in request:
            indexers = request["self"]["indexers"]
        elif "indexers" in request:
            indexers = request["indexers"]

        rolling = data.rolling(dim={dim: rate}, min_periods=rate, stride=rate)
        x_ds = rolling.max().dropna(dim)

        if "sampling_rate" in data.attrs:
            x_ds.attrs["sampling_rate"] = data.attrs["sampling_rate"] / rate

        if indexers is not None:

            try:
                i2slices = indexers_to_slices(indexers)
                x_ds = x_ds.sel(i2slices)
            except Exception as ex:
                print(x_ds, x_ds["time"].values, i2slices, indexers, request)
                not_null = pd.notnull(x_ds[dim])
                x_ds = x_ds.isel({dim: not_null})
                i2slices = indexers_to_slices(indexers)
                x_ds = x_ds.sel(i2slices)

            x_ds = exclusive_indexing(x_ds, indexers)

        return x_ds


class LTTBDownsampling(Node):
    def __init__(self, rate=1, dim=None):
        # Based on and many thanks to https://github.com/javiljoen/lttb.py
        super().__init__(rate=rate, dim=dim)

    def forward(self, data=None, request=None):
        dim = request["dim"]
        rate = request["rate"]
        if dim is None:
            dim = data.dims[-1]
        """ The following tries to re-implement the LTTB algorithm for xarray
            There are several issues:
            1. We cannot use the LTTB package
               it expects the index and the data in one numpy array. Since our index is
               usually in a xarray coordinate, we would need to disassemble the xarray
               and then reassemble it. (As done in the current implementation).
               We would potentially loose some information from the original datastructure
               and we would need to assume certain datatypes for certain dimensions.
            2. For multi-dimensional arrays, we run into issues since the dimension
               coordinate we want to reduce on applies to multiple axes and on each axis
               LTTB chooses a different pair of (index, value). Solution would be to choose
               the index for each window statically but choose the value depending on LTTB.
        """

        """ Note: The lttb implementation differs from our implementation
            due to different bucket generation. lttb uses numpy.array_split
            which subdivides into buckets of varying length ('almost' equal) and does not drop
            any values
            we use a rolling window which subdivides into equal lengths
            but drops values at the end

            form numpy docs for array_split:
            For an array of length l that should be split into n sections, it returns l % n sub-arrays of size l//n + 1 and the rest of size l//n.
        """

        # adapt downsampling rate
        n_out = data.sizes[dim] / rate
        rate = np.ceil(data.sizes[dim] / (n_out - 2)).astype(np.int)

        end = data.isel({dim: -1})
        data = data.isel({dim: slice(0, -1)})
        rolling = data.rolling(dim={dim: rate}, stride=rate)

        index_bins = (
            data[dim].rolling(dim={dim: rate}, stride=rate).construct("buckets")
        )

        value_bins = rolling.construct("buckets")

        value_mean = value_bins.mean("buckets")
        index_mean = index_bins.mean("buckets")
        # print(index_mean)
        out = []
        prev = None
        argmaxes = []
        for i in range(value_bins.sizes[dim]):
            current_value = value_bins.isel({dim: i})
            current_index = index_bins.isel({dim: i})
            # print(i,current_value.sizes['buckets'])
            # print('cv',current_value)
            if i == 0:
                # the first bucket consists of NaN and one entry
                # We choose the one entry and continue
                prev_value = current_value.isel(buckets=-1)
                prev_index = current_index.isel(buckets=-1)
                argmaxes.append(prev_value)
                continue

            # prev_data/time contains only one entry on `dim`-axis
            a = prev_value
            a_time = prev_index
            # current_data/time contains multiple entries on `dim`-axis
            bs = current_value
            bs_time = current_index
            if i < value_bins.sizes[dim] - 1:
                next_value = value_mean.isel({dim: i + 1})
                next_index = index_mean.isel({dim: i + 1})
            else:
                next_value = end
                next_index = end[dim]

            # calculate areas of triangle
            bs_minus_a = bs - a
            c_minus_p_value = current_value - prev_value
            c_minus_p_index = current_index - prev_index

            a_minus_bs = a - bs
            p_minus_c_value = prev_value - current_value
            p_minus_c_index = prev_index - current_index

            P_i, P_v = prev_index.astype(np.float), prev_value
            C_i, C_v = current_index.astype(np.float), current_value
            N_i, N_v = next_index.astype(np.float), next_value

            # print('P',P_i,P_v)
            # print('C',C_i,C_v)
            # print('N',N_i,N_v)
            # points P (prev), C (current), N (next)
            # Next is th emean
            # area = 0.5 * abs(P_i * (C_v - N_v) + C_i * (N_v - P_v) + N_i * (P_v - C_v))
            areas = 0.5 * abs(P_i * (C_v - N_v) + C_i * (N_v - P_v) + N_i * (P_v - C_v))

            # print(areas)
            # print(current_value)
            # print(current_value.shape)
            # print(areas.shape)
            # axis_num = current_value.get_axis_num('buckets')
            # arg = areas.argmax()
            # print('c',current_value)
            # print('oi',arg)
            try:
                arg = areas.argmax("buckets", skipna=False)
            except ValueError as ex:
                warnings.warn(ex)
                c = 0  # TODO: find a better solution to the ValueError("All-NaN slice encountered")
                # FIXME: arg is not correctly set and c will be overwritten in the failure case

            c = current_value.isel({"buckets": arg})
            # print(c)
            argmaxes.append(c)

        argmaxes.append(end)
        array = xr.concat(argmaxes, "time")
        # print(array)

        # TODO: integrate nicely
        use_pkg = False
        if use_pkg:
            import lttb

            # This is quite hacky and works only if the the selected dimension is a datetime
            if len(data.squeeze().dims) > 1:
                raise RuntimeError("Can only work with arrays of one dimension")

            d = np.array(
                [data["time"].values.astype(np.int64), data.squeeze().values]
            ).T
            small_data = lttb.downsample(d, n_out=n_out)

            array = xr.DataArray(
                small_data[:, 1],
                dims=[dim],
                coords={dim: (dim, pd.to_datetime(small_data[:, 0]))},
            )

            array.attrs = data.attrs
        return array
