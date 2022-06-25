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

import io
import pathlib
import warnings
from copy import deepcopy
from email.policy import default
from subprocess import call

import numpy as np
import obspy
import pandas as pd
import xarray as xr
from obspy import Stream, Trace, UTCDateTime
from obspy.clients.fdsn import Client
from tqdm import tqdm

import foreal
from foreal.config import get_setting
from foreal.convenience import indexers_to_slices
from foreal.core import Node


def convert_trace_to_data_array(trace, starttime):
    sampling_rate = trace.stats.sampling_rate
    timestamps = np.arange(0, len(trace.data)) * (1.0 / sampling_rate)
    # if starttime is None:
    #     starttime = pd.to_datetime(trace.stats.starttime.timestamp)
    timestamps = pd.to_datetime(starttime).tz_localize(None) + pd.to_timedelta(
        timestamps, "seconds"
    )
    da = xr.DataArray(trace.data, dims="time", coords={"time": timestamps})
    da.attrs["stats"] = trace.stats
    return da


def convert_stream_to_data_array(stream):
    stream.sort()
    sampling_rates = {int(trace.stats.sampling_rate) for trace in stream}
    if len(sampling_rates) != 1:
        raise RuntimeError("All channels must have the same sampling rate")

    start_times = np.array([trace.stats.starttime.timestamp for trace in stream])
    end_times = np.array([trace.stats.endtime.timestamp for trace in stream])
    starttime, endtime = UTCDateTime(start_times.min()), UTCDateTime(end_times.max())
    stream = stream.trim(starttime=starttime, endtime=endtime)
    stream.merge()

    traces_list, stats_dict = [], {}
    for trace in stream:
        trace_id = trace.id
        da = convert_trace_to_data_array(trace, starttime.datetime)
        da.name = trace.stats["channel"]
        traces_list.append(da)
        stats_dict[trace_id] = da.attrs["stats"]
    traces = xr.merge(traces_list, join="outer", compat="no_conflicts")
    traces_da = traces.to_array(dim="channel")
    traces_da.attrs["stats"] = stats_dict
    return traces_da


def obspy2xarray(stream):
    if isinstance(stream, Trace):
        stream = Stream(traces=[stream])
    if not isinstance(stream, Stream):
        raise RuntimeError("obspy2xarray can only convert obspy streams")

    if not len(stream):
        raise RuntimeError("No data in stream")
    da = convert_stream_to_data_array(stream)
    da.name = "stream"
    sampling_rate = stream[0].stats.sampling_rate

    da.attrs["sampling_rate"] = sampling_rate
    return da


def default_get_obspy_stream(
    store,
    request,
    get_filename=None,
    block_duration="1 day",
    pad=False,
    fill=None,
    extension="",
):
    start_time = request["indexers"]["time"]["start"]
    end_time = request["indexers"]["time"]["stop"]
    network = request["network"]
    stations = request["station"]
    channels = request["channel"]
    location = request["location"]
    extension = request.get("extension", extension)

    if not isinstance(channels, list):
        channels = [channels]
    if not isinstance(stations, list):
        stations = [stations]

    if extension != "" and extension[0] != ".":
        extension = "." + extension

    block_duration = pd.to_timedelta(block_duration)

    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    # We will get the full duration seismic data and trim it to the desired length afterwards
    start_time_floored = start_time.floor(freq=block_duration)
    timerange = pd.date_range(
        start=start_time_floored, end=end_time, freq=block_duration
    )
    stream = obspy.Stream()

    # loop through all hours
    for i in range(len(timerange)):
        h = timerange[i]

        st_list = obspy.Stream()

        filenames = {}

        for station in stations:
            for channel in channels:
                # Load either from store or from filename
                # get the file relative to the store
                filename = get_filename(
                    network,
                    station,
                    location,
                    channel,
                    timerange[i],
                    timerange[i] + block_duration,
                    extension,
                )
                filenames[channel] = filename
                try:
                    st = obspy.read(io.BytesIO(store[str(filename)]))
                except Exception as ex:
                    print(f"SEISMIC SOURCE Exception: {ex}")  # pass
                    st = obspy.Stream()
                st_list += st

        stream_h = st_list.merge(method=0, fill_value=fill)
        segment_h = stream_h

        stream += segment_h

    if not stream:
        raise RuntimeError(f"files not found {[filenames[fn] for fn in filenames]}")

    stream = stream.merge(method=0, fill_value=fill)

    stream = stream.trim(
        starttime=obspy.UTCDateTime(start_time),
        pad=pad,
        fill_value=fill,
        nearest_sample=True,
    )
    stream = stream.trim(
        endtime=obspy.UTCDateTime(end_time),
        pad=pad,
        fill_value=fill,
        nearest_sample=False,
    )

    stream.sort(
        keys=["channel"]
    )  # TODO: change this so that the order of the input channels list is maintained

    return stream


class SeismicPortal(Node):
    def __init__(
        self,
        store=None,
        station=None,
        channel=None,
        indexers=None,
        use_arclink=False,
        return_obspy=False,
        obspy_transform=None,
        get_obspy_stream=None,
        get_filename=None,
        extension="",
        **kwargs,
    ):  # TODO: update description
        """Seismic data source to get data
            The user can predefine the source's settings or provide them in a request
            Predefined setting should never be updated (to stay thread safe), but will be ignored
            if request contains settings

        Keyword Arguments:
            use_arclink {bool}  -- If true, downloads the data from the arclink service (authentication required) (default: {False})
            return_obspy {bool} -- By default an xarray is returned. If true, an obspy stream will be returned (default: {False})
        """

        # Make sure default arguments are non-mutable
        if indexers is None:
            indexers = {}

        super().__init__(
            station=station,
            channel=channel,
            indexers=indexers,
            use_arclink=use_arclink,
            return_obspy=return_obspy,
            extension=extension,
            kwargs=kwargs,
        )
        self.store = store
        if isinstance(self.store, str) or isinstance(self.store, pathlib.Path):
            self.store = foreal.DirectoryStore(self.store)

        self.obspy_transform = obspy_transform

        self.get_obspy_stream = get_obspy_stream
        self.get_filename = get_filename

    def __dask_tokenize__(self):
        return (SeismicPortal,)

    def configure(self, requests=None):
        """Default configure for DataSource nodes
        Arguments:
            request {list} -- List of requests

        Returns:
            dict -- Original request or merged requests
        """
        request = super().configure(requests)  # merging request here

        new_request = {}
        new_request.update(request)

        return new_request

    def forward(self, data=None, request=None):
        r = dict(request["self"])

        for item in ["network", "location", "station", "channel"]:
            if item in r.get("indexers", {}):
                r[item] = r["indexers"][item]
        network = r.get("network", "")
        location = r.get("location", "")
        station = r.get("station", "")

        if r["use_arclink"]:

            if isinstance(r["use_arclink"], dict):
                arclink = r["use_arclink"]
            else:
                try:
                    arclink = get_setting("arclink")
                except KeyError as err:
                    raise RuntimeError(
                        f"The following error occurred \n{err}. "
                        "Please provide either the credentials to access arclink or a path to the dataset"
                    )

            arclink_url = arclink.get("url", "")  # e.g. "http://arclink.ethz.ch"
            arclink_user = arclink.get("user", None)
            arclink_password = arclink.get("password", None)
            fdsn_client = Client(
                base_url=arclink_url,
                user=arclink_user,
                password=arclink_password,
            )

            if isinstance(r["channel"], list):
                channel = ",".join(r["channel"])
            else:
                channel = r["channel"]

            if isinstance(station, list):
                fdsn_station = ",".join(station)
            if isinstance(location, list):
                fdsn_location = ",".join(location)
            if isinstance(network, list):
                fdsn_network = ",".join(network)

            try:
                x = fdsn_client.get_waveforms(
                    network=fdsn_network,
                    station=fdsn_station,
                    location=fdsn_location,
                    channel=channel,
                    starttime=UTCDateTime(r["indexers"]["time"]["start"]),
                    endtime=UTCDateTime(r["indexers"]["time"]["stop"]),
                    attach_response=True,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Could not load waveform from fdsn for the request: {r}"
                )

            # TODO: potentially resample

        else:

            # logging.info('Loading seismic with fdsn')

            if self.get_obspy_stream is None:
                get_obspy_stream = default_get_obspy_stream
                if self.get_filename is None:
                    raise RuntimeError(
                        "Provide a `get_filename` method to load custom data from disk"
                    )
            else:
                get_obspy_stream = self.get_obspy_stream

            x = get_obspy_stream(
                self.store,
                r,
                get_filename=self.get_filename,
            )

        if "time" in r["indexers"]:
            # make the traces in the stream have a common start and end time
            x = x.slice(
                starttime=UTCDateTime(r["indexers"]["time"]["start"]),
                #            nearest_sample=True, #why was it set to true?
                nearest_sample=False,
            )
            x = x.slice(
                endtime=UTCDateTime(r["indexers"]["time"]["stop"]),
                nearest_sample=False,
            )

        if callable(self.obspy_transform):
            # if "obspy_transform" in r and callable("obspy_transform"):
            # excpects a callable which input is an obspy stream
            # and returns and obspy stream
            x = self.obspy_transform(x)

        if not r["return_obspy"]:
            obs = x
            da = obspy2xarray(x)
            # print('load',da)
            # del da.attrs["stats"]

            da = da.expand_dims("station", 0)
            if not isinstance(station, list):
                station = [station]
            da = da.assign_coords({"station": station})

            da = da.expand_dims("network", 0)
            if not isinstance(network, list):
                network = [network]
            da = da.assign_coords({"network": network})

            da = da.expand_dims("location", 0)
            if not isinstance(location, list):
                location = [location]
            da = da.assign_coords({"location": location})

            da = da.astype(np.float, casting="safe")
            x = da

            # TODO: same check for obspy
            if "channel" in r and r["channel"] is not None:
                if len(x["channel"]) != len(r["channel"]):
                    warnings.warn("Inconsistent data: Not all channels could be loaded")

            x["channel"] = x["channel"].astype("str")

            # round it to the sampling rate
            steps_in_ns = 1 / x.attrs["sampling_rate"] * 1000 * 1000 * 1000
            # x['time'] = x['time'].dt.round(freq=f'{int(steps_in_ns)}N')
            x["time"] = x["time"].dt.floor(freq=f"{int(steps_in_ns)}N")

            # NOTE: we are faking exclusive slicing, since xarray (or pandas beneath) is always inclusive
            end_time = x["time"].max() - pd.to_timedelta(
                1 / x.attrs["sampling_rate"], "seconds"
            )
            x = x.sel({"time": slice(None, end_time)})

            slices = indexers_to_slices(r["indexers"])
            if "time" in slices:
                del slices["time"]

            x = x.sel(slices)

        return x
