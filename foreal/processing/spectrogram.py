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

import numpy as np
import pandas as pd
import scipy.signal
import xarray as xr

from foreal.config import get_setting
from foreal.convenience import dict_update, indexers_to_slices, to_datetime
from foreal.core import Node


class Spectrogram(Node):
    def __init__(
        self,
        nfft=2048,
        stride=1024,
        dim=None,
        sampling_rate=None,
        window=("tukey", 0.25),
        detrend="constant",
        return_onesided=True,
        scaling="density",
        mode="psd",
    ):
        super().__init__(
            nfft=nfft,
            stride=stride,
            dim=dim,
            sampling_rate=sampling_rate,
            window=window,
            detrend=detrend,
            return_onesided=return_onesided,
            scaling=scaling,
            mode=mode,
        )
        # TODO: make generic
        self.reference_value = to_datetime("2000-01-01", format="%Y-%m-%d").tz_localize(
            None
        )

    def align_indexers(
        self, index, stride, sampling_rate, round_type="floor", end_index=None
    ):
        region = stride / sampling_rate  # seconds
        region = pd.to_timedelta(region, "seconds")
        # FIXME: make this index type dependant
        if isinstance(index, str):
            index = to_datetime(index)
        #            index = pd.to_datetime(index,format=get_setting('datetime_format'),exact=False)
        #            index = pd.to_datetime(index,infer_datetime_format=True)
        chunk_index = (
            np.floor((index - self.reference_value) / region) * region
            + self.reference_value
        )

        if end_index is not None:
            return

        if round_type == "ceil":
            chunk_index += region

        return chunk_index

    def aligned_range(self, start, stop, stride, sampling_rate, stop_type="dim"):
        aligned_start = self.align_indexers(start, stride, sampling_rate)
        if stop_type == "index":
            region = stride / sampling_rate  # seconds
            region = pd.to_timedelta(region, "seconds")
            aligned_stop = (
                self.align_indexers(start, stride, sampling_rate) + stop * region
            )
        else:
            aligned_stop = self.align_indexers(stop, stride, sampling_rate)

        return aligned_start, aligned_stop

    def configure(self, requests):
        """

        Arguments:
            request {list} -- List of requests

        Returns:
            dict -- Original, updated or merged request(s)
        """
        requests = super().configure(requests)

        requests = deepcopy(dict(requests))
        rs = requests["self"]
        indexers = requests["indexers"]
        # print('r',requests)
        if rs["dim"] != "time":
            raise NotImplementedError()

        if rs["dim"] not in indexers:
            return requests

        requests["indexers"][rs["dim"]]["start"] = to_datetime(
            requests["indexers"][rs["dim"]]["start"]
        )
        requests["indexers"][rs["dim"]]["stop"] = to_datetime(
            requests["indexers"][rs["dim"]]["stop"]
        )

        if rs.get("sampling_rate", None) is not None:
            window_length_seconds = pd.to_timedelta(
                rs["nfft"] / rs["sampling_rate"], "seconds"
            )

            # dict_update(rs,{"indexers": deepcopy(requests["indexers"])},True)
            requests["indexers"][rs["dim"]]["start"] = (
                requests["indexers"][rs["dim"]]["start"] - window_length_seconds
            )
            requests["indexers"][rs["dim"]]["stop"] = (
                requests["indexers"][rs["dim"]]["stop"] + window_length_seconds
            )

        if rs["dim"] in indexers:
            if (
                rs.get("sampling_rate", None) is not None
                and rs.get("stride", None) is not None
            ):
                # align the start of the indexing to make the spectogram windows evenly distributed
                requests["indexers"][rs["dim"]]["start"] = self.align_indexers(
                    requests["indexers"][rs["dim"]]["start"],
                    rs["stride"],
                    rs["sampling_rate"],
                )
                requests["indexers"][rs["dim"]]["stop"] = self.align_indexers(
                    requests["indexers"][rs["dim"]]["stop"],
                    rs["stride"],
                    rs["sampling_rate"],
                    round_type="ceil",
                )

        requests["indexers"][rs["dim"]]["start"] = requests["indexers"][rs["dim"]][
            "start"
        ].isoformat()
        requests["indexers"][rs["dim"]]["stop"] = requests["indexers"][rs["dim"]][
            "stop"
        ].isoformat()

        requests["requires_request"] = True
        return requests

    def forward(self, data=None, request=None):
        # config = deepcopy(self.config)
        # if request is not None:
        #     config.update(request)
        config = request["self"]

        # if 'time' in request["self"]["indexers"]:
        #     start_time = pd.to_datetime(
        #         request["self"]["indexers"]["time"]["start"]
        #     )
        #     stop_time = pd.to_datetime(
        #         request["self"]["indexers"]["time"]["stop"]
        #     )

        if config["dim"] is None:
            config["dim"] = data.dims[-1]

        if "sampling_rate" not in data.attrs:
            if config["sampling_rate"] is None:
                raise RuntimeError(
                    "Please provide a sampling_rate attribute "
                    "to your config or your input data"
                )
        else:
            config["sampling_rate"] = data.attrs["sampling_rate"]

        if "raw" in config:
            samples = data
            axis = -1
        else:
            samples = data.data
            axis = data.get_axis_num(config["dim"])
        noverlap = config["nfft"] - config["stride"]

        freqs, spectrum_time, spectrum = scipy.signal.spectrogram(
            samples,
            nfft=config["nfft"],
            nperseg=config["nfft"],
            noverlap=noverlap,
            fs=config["sampling_rate"],
            axis=axis,
            detrend=config["detrend"],
            scaling=config["scaling"],
            return_onesided=config["return_onesided"],
            mode=config["mode"],
            window=config["window"],
        )
        if "raw" in config:
            diff_start = to_datetime(
                request["self"]["indexers"]["time"]["start"]
            ) - to_datetime(request["indexers"]["time"]["start"])
            diff_stop = to_datetime(
                request["self"]["indexers"]["time"]["stop"]
            ) - to_datetime(request["indexers"]["time"]["start"])
            x = (
                pd.TimedeltaIndex(pd.to_timedelta(spectrum_time, "s"))
                .get_indexer([diff_start, diff_stop])
                .tolist()
            )
            spectrum = spectrum[..., x[0] : x[1]]
            return spectrum

        if len(data[config["dim"]]) == 0:
            raise RuntimeError(
                "Cannot compute Spectrogram for data with shape", data.shape
            )
        else:
            # TODO: check if this is what we want. it's: the earliest timestamp of input + the delta computed by scipy
            ds_coords = to_datetime(
                to_datetime(data[config["dim"]].min().values)
                + pd.to_timedelta(spectrum_time, "s")
            ).tz_localize(
                None
            )  # TODO: change when xarray #3291 is fixed

        # Create a new DataArray for the spectogram
        dims = (
            data.dims[:axis]
            + ("frequency",)
            + data.dims[(axis + 1) :]
            + (config["dim"],)
        )
        coords = dict(data.coords)
        coords.update({"frequency": ("frequency", freqs), config["dim"]: ds_coords})

        dataarray = xr.DataArray(spectrum, dims=dims, coords=coords)
        # print(dataarray,indexers_to_slices(request["self"]["indexers"]))
        # print(pd.isnull(dataarray['time']).any())
        try:
            dataarray.attrs["sampling_rate"] = 1 / (spectrum_time[1] - spectrum_time[0])
        except Exception as e:
            pass

        slices = indexers_to_slices(request["self"]["indexers"])
        # dataarray = dataarray.sel(slices)
        dataarray["time"] = dataarray["time"].to_index()
        #        print(dataarray,slices)
        dataarray = dataarray.loc[slices]
        return dataarray
