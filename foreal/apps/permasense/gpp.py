# -*- coding: utf-8 -*
import datetime
import html
import pathlib
import urllib.request
import warnings
from copy import deepcopy
from datetime import timezone

import numpy as np
import obspy
import pandas as pd
import xarray
from obspy.core.trace import Trace

import foreal
from foreal.apps.seismic import obspy2xarray
from foreal.config import get_setting
from foreal.convenience import indexers_to_slices
from foreal.core import Node

# used to take into account possible overlap from previous days
MAX_EVENT_DURATION = datetime.timedelta(hours=1)


class SeismicGPP(Node):
    def __init__(
        self,
        deployment=None,
        position=None,
        event_data_dir=None,
        with_data=True,
        return_obspy=False,
        stats_template={"network": "XX", "location": "", "station": "DEGGPP3x"},
        **kwargs,
    ):
        ## written by Tobias Kuonen modified by Matthias Meyer
        if event_data_dir is None:
            event_data_dir = (
                pathlib.Path(get_setting("permasense_vault_dir"))
                / "datasets"
                / "gpp_data"
            )
        super().__init__(
            deployment=deployment,
            position=position,
            event_data_dir=event_data_dir,
            with_data=with_data,
            return_obspy=False,
            **kwargs,
        )

        self.stats_template = stats_template

    def scale_value(self, value, adc_pga, bytes_per_sample):
        """Converts integer values from the integer output of the ADC to mV

        Arguments:
            value -- the integer value to convert
            adc_pga -- The pre amplification factor of the ADC
            pytesPerSample: -- The number of bytes each sample uses

        Returns:
            convertedValue -- The converted value

        """
        # Inspired by: R-Script: permasense_geophone_exporter.R (author: Jan Beutel)
        return value - 2 ** ((bytes_per_sample * 8) - 1)
        return (value * 2500 / (2 ** (bytes_per_sample * 8) - 1) - 1250) / adc_pga

    def convert_binary_data(self, binary_data, nsamples, adc_pga, endian="big"):
        """Converts the binary output of the ADC to a list of samples in mV

        Argumens:
            binary_data {bytes} -- the binary input to convert
            nsamples -- number of samples
            adc_pga -- The pre amplification factor of the ADC
            edian -- the byteorder: 'big' or 'little'; Default: 'big'

        Returns:
            result -- list of samples in mV

        """
        # inspired by script adcdata_to_csv.py by rdaforno
        # TODO: assert (len(binary_data)/nsamples %1 == 0
        if nsamples == 0:
            return []
        bytes_per_sample = int(len(binary_data) / nsamples)
        result = np.zeros(nsamples)
        for i in range(0, nsamples):
            result[i] = self.scale_value(
                int.from_bytes(
                    binary_data[(bytes_per_sample * i) : (bytes_per_sample * (i + 1))],
                    byteorder=endian,
                    signed=False,
                ),
                adc_pga,
                bytes_per_sample,
            )
        return result

    def get_gsn_url(self, vsensor, conditions):
        """Create url to retrieve GSN data according to a variable number of
        conditions

        Arguments:
            vsensor -- Name of virtual sensor
            conditions -- list of conditions in the form: [{
                            'join':<'and' or 'or'>,'field':<name of field>,
                            'min':<min value>,
                            'max':<max value>}]

        Returns:
            url -- the assembled url

        """
        # Documentation of url format:
        # https://github.com/LSIR/gsn/wiki/Web-Interface
        # https://www.earth-syst-sci-data.net/11/1203/2019/
        # Inspired by: GSN Data Manager in old permasense repository
        url = (
            get_setting("permasense_server")
            + "/multidata?field[0]=All&vs[0]="
            + vsensor
        )
        for i in range(0, len(conditions)):
            condition = conditions[i]
            condition["i"] = i
            condition["vs"] = vsensor
            url += (
                "&c_join[{i:d}]={join:s}&c_vs[{i:d}]={vs:s}"
                + "&c_field[{i:d}]={field:s}&c_min[{i:d}]={min:d}"
                + "&c_max[{i:d}]={max:d}"
            ).format(**condition)
        return url

    @staticmethod
    def get_gsn_adc_data(path):
        """Download binary samples form from the permasense server and return
        them

        Arguments:
            path -- path of the file on the permasense_server (html escaped)

        Result:
            data -- the data retrieved from that url or [] if the return code
                    of the server wasn't 200
        """

        # make http request
        request = urllib.request.urlopen(
            get_setting("permasense_server") + html.unescape(path)
        )
        # if return code not ok: print error and return empty list
        if request.getcode() != 200:
            print(
                "Error while downloading binary samples of Event-Data from "
                + str(get_setting("permasense_server"))
            )
            return []
        # if no error: return retrieved data
        return request.read()

    def convert_metadata(self, metadata):
        for element in [
            ["generation_time", 1000],
            ["timestamp", 1000],
            ["start_time", 1000000],
            ["first_time", 1000000],
            ["end_time", 1000000],
            ["generation_time_microsec", 1000000],
        ]:
            if element[0] in metadata and not metadata[element[0]] is None:
                metadata[element[0]] = datetime.datetime.fromtimestamp(
                    metadata[element[0]] / element[1], tz=timezone.utc
                ).replace(tzinfo=None)

        # Some metadata fileds have different names in the GSN and in the
        # permasense_vault:  Rename such fields to have consistent naming:
        metadata.rename(
            {
                "adc_sps": "sampling_rate",
                "adc_sampling_freq": "sampling_rate",
                "ID": "id",
                "end_time": "gsn_end_time",
                "samples": "total_samples",
            },
            inplace=True,
        )

        if "channels" not in metadata:
            metadata["channels"] = "Z;E;N"

        metadata["channels"] = metadata["channels"].split(";")
        metadata["channels"] = ["EH" + c for c in metadata["channels"]]

        # old it's called samples, new it's called total_samples
        metadata["samples"] = metadata["total_samples"] // len(metadata["channels"])

        # add end time of event to the metadata:
        metadata["end_time"] = metadata["first_time"] + datetime.timedelta(
            seconds=(metadata["samples"] - 1) / metadata["sampling_rate"]
        )
        metadata["adc_pga"] = max(1, metadata["adc_pga"])
        # Convert the values of the max and min peak:
        metadata["peak_pos_val"] = self.scale_value(
            metadata["peak_pos_val"], metadata["adc_pga"], 3
        )
        metadata["peak_neg_val"] = self.scale_value(
            metadata["peak_neg_val"], metadata["adc_pga"], 3
        )
        return metadata

    def integrity():
        pass

    def get_data(
        self,
        deployment,
        position,
        start_time,
        end_time,
        event_data_dir,
        with_data=True,
        channels=None,
    ):
        """returns all events which intersect with the interval
        [start_time,end_time]

        Arguments:
            start_time {datetime.datetime} -- start time of interval (inclusive)
            end_time {datetime.datetime} -- end time of interval (inclusive)
        Returns:
            result -- the requested events; format:
                          XArray of {'Event':<Xarray of Event>}
        """
        results = []  # array of {'Event:'Xarray}
        added_events = []

        if not event_data_dir.exists():
            print("Event data source: Warning: event_data_dir does not exist.")

        else:
            stream = obspy.Stream()

            current_date = (start_time - MAX_EVENT_DURATION).date()
            while current_date <= end_time.date():
                current_date_dir = (
                    event_data_dir
                    / deployment
                    / str(position)
                    / current_date.strftime("%Y-%m-%d")
                )
                if current_date_dir.exists():
                    # for current_dir in current_date_dir.iterdir():
                    current_dir = current_date_dir
                    event_list_file = current_dir / "_ACQ.TXT"
                    if not event_list_file.exists():
                        event_list_file = current_dir / "_ACQ.CSV"
                    if event_list_file.exists():
                        event_list = pd.read_csv(event_list_file, dtype={"ID": str})
                        for index, event in event_list.iterrows():
                            metadata = self.convert_metadata(event)

                            if (
                                metadata["end_time"] < start_time
                                or metadata["start_time"] > end_time
                            ):
                                continue

                            data = []
                            time = np.array([], dtype="datetime64[ns]")
                            if with_data:
                                traces = []
                                if channels is None:
                                    channels_to_load = metadata["channels"]
                                else:
                                    channels_to_load = channels

                                for channel in channels_to_load:
                                    stats = deepcopy(self.stats_template)

                                    data_file_path = current_dir / (
                                        f"{event['id']}.{channel[-1]}.MSEED"
                                    )

                                    if data_file_path.exists():
                                        # it's a miniseed file
                                        xtrace = obspy.read(str(data_file_path))[0]
                                    else:
                                        # it's not a miniseed but raw format
                                        data_file_path = current_dir / (
                                            f"{event['id']}.{channel[-1]}.DAT"
                                        )
                                        if not data_file_path.exists():
                                            channel2idx = {"EHZ": 1, "EHE": 2, "EHN": 3}
                                            channel_idx = channel2idx[channel]
                                            data_file_path = current_dir / (
                                                f"{event['id']}A{channel_idx}.DAT"
                                            )

                                        if not data_file_path.exists():
                                            print(
                                                "Warning: Missing Data: File "
                                                + str(data_file_path)
                                                + " not found."
                                            )
                                            continue
                                        data_file = data_file_path.open(mode="rb")
                                        binary_data = data_file.read()
                                        data_file.close()
                                        data = self.convert_binary_data(
                                            binary_data,
                                            metadata["samples"],
                                            metadata["adc_pga"],
                                        )
                                        xtrace = Trace(data)

                                        stats_dynamic = {
                                            "sampling_rate": metadata["sampling_rate"],
                                            "starttime": metadata["first_time"],
                                        }
                                        stats.update(stats_dynamic)

                                    xtrace.stats.update(stats)
                                    xtrace.stats.update({"channel": channel})

                                    stream += xtrace

                current_date += datetime.timedelta(days=1)

        if not stream:
            raise RuntimeError(f"files not found")

        stream = stream.merge()

        stream = stream.trim(
            starttime=obspy.UTCDateTime(start_time),
            # pad=pad,
            # fill_value=fill,
            nearest_sample=True,
        )
        stream = stream.trim(
            endtime=obspy.UTCDateTime(end_time),
            # pad=pad,
            # fill_value=fill,
            nearest_sample=False,
        )
        return stream

    def configure(self, requests):
        return super().configure(requests)

    def forward(self, data=None, request=None):
        r = request["self"]
        start_time = pd.to_datetime(r["indexers"]["time"]["start"])
        end_time = pd.to_datetime(r["indexers"]["time"]["stop"])
        x = self.get_data(
            r["deployment"],
            r["position"],
            start_time,
            end_time,
            r["event_data_dir"],
            with_data=r["with_data"],
            channels=r["indexers"].get("channel", None),
        )
        if "obspy_transform" in r and callable("obspy_transform"):
            # excpects a callable which input is an obspy stream
            # and returns and obspy stream
            x = r["obspy_transform"](x)

        if not r["return_obspy"]:
            obs = x
            station = x[0].stats["station"]
            network = x[0].stats["network"]
            location = x[0].stats["location"]
            da = obspy2xarray(x)
            del da.attrs["stats"]

            da = da.expand_dims("station", 0)
            da = da.assign_coords({"station": [station]})
            da = da.expand_dims("network", 0)
            da = da.assign_coords({"network": [network]})
            da = da.expand_dims("location", 0)
            da = da.assign_coords({"location": [location]})
            da = da.astype(np.float, casting="safe")
            x = da

            # TODO: same check for obspy
            if "channel" in r and len(x["channel"]) != len(r["channel"]):
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
            del slices["time"]
            x = x.sel(slices)
        return da
