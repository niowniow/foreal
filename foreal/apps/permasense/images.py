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

import base64
import datetime as dt
import io
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
from tqdm import tqdm

from foreal import DirectoryStore
from foreal.config import get_setting, set_setting, setting_exists
from foreal.convenience import read_csv_with_store, to_csv_with_store, to_datetime
from foreal.core import Node


class MHDSLRFilenames(Node):
    def __init__(
        self,
        base_directory=None,
        store=None,
        method="directory",
        indexers=None,
        force_write_to_remote=False,
        as_pandas=True,
    ):
        """Fetches the DSLR images from the Matterhorn deployment, returns the image
            filename(s) corresponding to the end and start time provided in either the
            config dict or as a request to the __call__() function.

        Arguments:
            StuettNode {[type]} -- [description]  # FIXME: Incorrect doc string

        Keyword Arguments:
            base_directory {[type]} -- [description]
            method {str}     -- [description] (default: {'directory'})
        """
        if indexers is None:
            indexers = {}

        if store is None and base_directory is not None:
            store = DirectoryStore(base_directory)

        super().__init__(
            base_directory=base_directory,
            store=store,
            method=method,
            indexers=indexers,
            force_write_to_remote=force_write_to_remote,
            as_pandas=as_pandas,
        )

    def configure(self, requests):
        request = super().configure(requests)
        new_request = {"requires_request": True}
        new_request.update(request)
        return new_request

    def forward(self, data=None, request=None):
        """Retrieves the images for the selected time period from the server. If only a start_time timestamp is provided,
          the file with the corresponding date will be loaded if available. For periods (when start and end time are given)
          all available images are indexed first to provide an efficient retrieval.

        Arguments:
            start_time {datetime} -- If only start_time is given the neareast available image is return. If also end_time is provided the a dataframe is returned containing image filenames from the first image after start_time until the last image before end_time.

        Keyword Arguments:
            end_time {datetime} -- end time of the selected period. see start_time for a description. (default: {None})

        Returns:
            dataframe -- Returns containing the image filenames of the selected period.
        """
        config = request
        methods = ["directory", "web"]
        if request["self"]["method"].lower() not in methods:
            raise RuntimeError(
                f"The {request['self']['method']} output_format is not supported. Allowed formats are {methods}"
            )

        if (
            request["self"]["base_directory"] is None
            and request["self"]["store"] is None
        ) and request["self"]["method"].lower() != "web":
            raise RuntimeError("Please provide a base_directory containing the images")

        if request["self"]["method"].lower() == "web":  # TODO: implement
            raise NotImplementedError("web fetch has not been implemented yet")

        # print(request)
        try:
            start_time = pd.to_datetime(request["indexers"]["time"]["start"])
        except KeyError:
            start_time = None
        try:
            end_time = pd.to_datetime(request["indexers"]["time"]["stop"])
        except KeyError:
            end_time = None

        if start_time is None:
            raise RuntimeError("Please provide at least a start time")

        # if it is not timezone aware make it
        # if start_time.tzinfo is None:
        #     start_time = start_time.replace(tzinfo=timezone.utc)

        # If there is no tmp_dir we can try to load the file directly, otherwise
        # there will be a warning later in this function and the user should
        # set a tmp_dir
        if end_time is None and (
            not setting_exists("user_dir") or not os.path.isdir(get_setting("user_dir"))
        ):
            image_filename = self.get_image_filename(start_time)
            if image_filename is not None:
                return image_filename

        # If we have already loaded the dataframe in the current session we can use it
        if setting_exists("image_list_df"):
            imglist_df = get_setting("image_list_df")
        else:
            filename = "image_integrity.csv"
            success = False
            # first try to load it from remote via store
            if request["self"]["store"] is not None:
                if filename in request["self"]["store"]:
                    imglist_df = read_csv_with_store(request["self"]["store"], filename)
                    success = True
                elif request["self"]["force_write_to_remote"]:
                    # try to reload it and write to remote
                    imglist_df = self.image_integrity_store(request["self"]["store"])
                    try:
                        to_csv_with_store(
                            request["self"]["store"],
                            filename,
                            imglist_df,
                            dict(index=False),
                        )
                        success = True
                    except Exception as e:
                        print(e)

            # Otherwise we need to load the filename dataframe from disk
            if (
                not success
                and setting_exists("user_dir")
                and os.path.isdir(get_setting("user_dir"))
            ):
                imglist_filename = os.path.join(get_setting("user_dir"), "") + filename

                # If it does not exist in the temporary folder of our application
                # We are going to create it
                if os.path.isfile(imglist_filename):
                    # imglist_df = pd.read_parquet(
                    #     imglist_filename
                    # )  # TODO: avoid too many different formats
                    imglist_df = pd.read_csv(imglist_filename)
                else:
                    # we are going to load the full list => no arguments
                    imglist_df = self.image_integrity_store(request["self"]["store"])
                    # imglist_df.to_parquet(imglist_filename)
                    imglist_df.to_csv(imglist_filename, index=False)
            elif not success:
                # if there is no tmp_dir we can load the image list but
                # we should warn the user that this is inefficient
                imglist_df = self.image_integrity_store(request["self"]["store"])
                warn(
                    "No temporary directory was set. You can speed up multiple runs of your application by setting a temporary directory"
                )

            # TODO: make the index timezone aware
            imglist_df.set_index("start_time", inplace=True)
            imglist_df.index = to_datetime(imglist_df.index)
            imglist_df.sort_index(inplace=True)

            set_setting("image_list_df", imglist_df)

        output_df = None
        if end_time is None:
            if start_time < imglist_df.index[0]:
                start_time = imglist_df.index[0]

            loc = imglist_df.index.get_loc(start_time, method="nearest")
            output_df = imglist_df.iloc[loc : loc + 1]
        else:
            # if end_time.tzinfo is None:
            #     end_time = end_time.replace(tzinfo=timezone.utc)
            if start_time > imglist_df.index[-1] or end_time < imglist_df.index[0]:
                # return empty dataframe
                output_df = imglist_df[0:0]
            else:
                if start_time < imglist_df.index[0]:
                    start_time = imglist_df.index[0]
                if end_time > imglist_df.index[-1]:
                    end_time = imglist_df.index[-1]

                output_df = imglist_df.iloc[
                    imglist_df.index.get_loc(
                        start_time, method="bfill"
                    ) : imglist_df.index.get_loc(end_time, method="ffill")
                    + 1
                ]

        if not request["self"]["as_pandas"]:
            output_df = output_df[["filename"]]  # TODO: do not get rid of end_time
            output_df.index.rename("time", inplace=True)
            # output = output_df.to_xarray(dims=["time"])
            output = xr.Dataset.from_dataframe(output_df).to_array()
            # print(output)
            # output = xr.DataArray(output_df['filename'], dims=["time"])
        else:
            output = output_df
        return output

    # TODO: write test for image_integrity_store
    def image_integrity_store(
        self, store, start_time=None, end_time=None, delta_seconds=0
    ):
        """Checks which images are available on the permasense server

        Keyword Arguments:
            start_time {[type]} -- datetime object giving the lower bound of the time range which should be checked.
                                   If None there is no lower bound. (default: {None})
            end_time {[type]} --   datetime object giving the upper bound of the time range which should be checked.
                                   If None there is no upper bound (default: {None})
            delta_seconds {int} -- Determines the 'duration' of an image in the output dataframe.
                                   start_time  = image_time+delta_seconds
                                   end_time    = image_time-delta_seconds (default: {0})

        Returns:
            DataFrame -- Returns a pandas dataframe with a list containing filename relative to self.base_directory,
                         start_time and end_time start_time and end_time can vary depending on the delta_seconds parameter
        """
        """ Checks which images are available on the permasense server

        Arguments:
            start_time:
            end_time:
            delta_seconds:  Determines the 'duration' of an image in the output dataframe.
                            start_time  = image_time+delta_seconds
                            end_time    = image_time-delta_seconds
        Returns:
            DataFrame --
        """
        if start_time is None:
            # a random year which is before permasense installation started
            start_time = dt.datetime(1900, 1, 1)
        if end_time is None:
            end_time = dt.datetime.utcnow()

        tbeg_days = start_time.replace(hour=0, minute=0, second=0)
        tend_days = end_time.replace(hour=23, minute=59, second=59)

        delta_t = dt.timedelta(seconds=delta_seconds)
        num_filename_errors = 0
        images_list = []

        print("Loading the foreal MHDSLR image store")
        for key in tqdm(store.keys()):
            try:
                pathkey = Path(key)
                datekey = pathkey.parent.name
                dir_date = pd.to_datetime(str(datekey), format="%Y-%m-%d")
            except Exception as ex:
                # we do not care for files not matching our format
                continue

            if pd.isnull(dir_date):
                continue

            # limit the search to the explicit time range
            if dir_date < tbeg_days or dir_date > tend_days:
                continue

            # print(file.stem)
            start_time_str = pathkey.stem
            try:
                _start_time = pd.to_datetime(start_time_str, format="%Y%m%d_%H%M%S")
                if start_time <= _start_time <= end_time:
                    images_list.append(
                        {
                            "filename": str(key),
                            "start_time": _start_time - delta_t,
                            "end_time": _start_time + delta_t,
                        }
                    )
            except ValueError:
                # FIXME: img_file does not exist
                # try old naming convention
                try:
                    _start_time = pd.to_datetime(start_time_str, format="%Y%m%d_%H%M%S")

                    if start_time <= _start_time <= end_time:
                        images_list.append(
                            {
                                "filename": str(key),
                                "start_time": _start_time - delta_t,
                                "end_time": _start_time + delta_t,
                            }
                        )
                except ValueError:
                    num_filename_errors += 1
                    warn(
                        "Permasense data integrity, the following is not a valid image filename and will be ignored: %s"
                        % str(key)
                    )
                    continue

        segments = pd.DataFrame(images_list)
        if not segments.empty:
            segments.drop_duplicates(inplace=True, subset="start_time")
            segments.start_time = to_datetime(segments.start_time)
            segments.end_time = to_datetime(segments.end_time)
            segments.sort_values("start_time")

        return segments

    def image_integrity(
        self, base_directory, start_time=None, end_time=None, delta_seconds=0
    ):
        store = DirectoryStore(base_directory)
        return self.image_integrity_store(store, start_time, end_time, delta_seconds)

    def get_image_filename(self, timestamp):
        """Checks whether an image exists for exactly the time of timestamp and returns its filename

            timestamp: datetime object for which the filename should be returned

        # Returns
            The filename if the file exists, None if there is no file
        """
        if self.config["base_directory"] is not None:
            datadir = self.config["base_directory"]
        elif self.config["store"] is not None:
            datadir = None
            store = self.config["store"]
        else:
            datadir = None

        new_filename = (
            timestamp.strftime("%Y-%m-%d")
            + "/"
            + timestamp.strftime("%Y%m%d_%H%M%S")
            + ".JPG"
        )
        old_filename = (
            timestamp.strftime("%Y-%m-%d")
            + "/"
            + timestamp.strftime("%Y-%m-%d_%H%M%S")
            + ".JPG"
        )
        if datadir is not None:
            if os.path.isfile(datadir + new_filename):
                return new_filename
            elif os.path.isfile(datadir + old_filename):
                return old_filename
            else:
                return None
        else:
            if new_filename in store:
                return new_filename
            elif old_filename in store:
                return old_filename
            else:
                return None

    def get_nearest_image_url(self, IMGparams, imgdate, floor=False):
        if floor:
            date_beg = imgdate - dt.timedelta(hours=4)
            date_end = imgdate
        else:
            date_beg = imgdate
            date_end = imgdate + dt.timedelta(hours=4)

        vs = []
        # predefine vs list
        field = []
        # predefine field list
        c_vs = []
        c_field = []
        c_join = []
        c_min = []
        c_max = []

        vs = vs + ["matterhorn_binary__mapped"]
        field = field + ["ALL"]
        # select only data from one sensor (position 3)
        c_vs = c_vs + ["matterhorn_binary__mapped"]
        c_field = c_field + ["position"]
        c_join = c_join + ["and"]
        c_min = c_min + ["18"]
        c_max = c_max + ["20"]

        c_vs = c_vs + ["matterhorn_binary__mapped"]
        c_field = c_field + ["file_complete"]
        c_join = c_join + ["and"]
        c_min = c_min + ["0"]
        c_max = c_max + ["1"]

        # create url which retrieves the csv data file
        url = "http://data.permasense.ch/multidata?"
        url = url + "time_format=" + "iso"
        url = url + "&from=" + date_beg.strftime("%d/%m/%Y+%H:%M:%S")
        url = url + "&to=" + date_end.strftime("%d/%m/%Y+%H:%M:%S")
        for i in range(0, len(vs), 1):
            url = url + "&vs[%d]=%s" % (i, vs[i])
            url = url + "&field[%d]=%s" % (i, field[i])

        for i in range(0, len(c_vs), 1):
            url = url + "&c_vs[%d]=%s" % (i, c_vs[i])
            url = url + "&c_field[%d]=%s" % (i, c_field[i])
            url = url + "&c_join[%d]=%s" % (i, c_join[i])
            url = url + "&c_min[%d]=%s" % (i, c_min[i])
            url = url + "&c_max[%d]=%s" % (i, c_max[i])

        url = url + "&timeline=%s" % ("generation_time")

        # print(url)
        d = pd.read_csv(url, skiprows=2)

        # print(d)

        # print(type(d['#data'].values))
        d["data"] = [s.replace("&amp;", "&") for s in d["data"].values]

        d.sort_values(by="generation_time")
        d["generation_time"] = to_datetime(d["generation_time"])

        if floor:
            data_str = d["data"].iloc[0]
            data_filename = d["relative_file"].iloc[0]
            # print(d['generation_time'].iloc[0])
            img_timestamp = d["generation_time"].iloc[0]
        else:
            data_str = d["data"].iloc[-1]
            data_filename = d["relative_file"].iloc[-1]
            # print(d['generation_time'].iloc[-1])
            img_timestamp = d["generation_time"].iloc[-1]

        file_extension = data_filename[-3:]
        base_url = "http://data.permasense.ch"
        # print(base_url + data_str)

        return base_url + data_str, img_timestamp, file_extension


class MHDSLRImages(MHDSLRFilenames):
    def __init__(
        self,
        base_directory=None,
        store=None,
        method="directory",
        output_format="xarray",
        image_processing=None,
        indexers=None,
    ):
        if indexers is None:
            indexers = {}

        if store is None and base_directory is not None:
            store = DirectoryStore(base_directory)
        self.image_processing = image_processing
        super().__init__(
            base_directory=None,
            store=store,
            method=method,
            indexers=indexers,
        )

        self.config["output_format"] = output_format

    def forward(self, data=None, request=None):
        filenames = super().forward(request=request)

        if request["self"]["output_format"] == "xarray":
            return self.construct_xarray(filenames)
        elif request["self"]["output_format"] == "base64":
            return self.construct_base64(filenames)
        else:
            output_formats = ["xarray", "base64"]
            raise RuntimeError(
                f"The {request['self']['output_format']} output_format is not supported. Allowed formats are {output_formats}"
            )

    def construct_xarray(self, filenames):
        images = []
        times = []
        channels = []
        for timestamp, element in filenames.iterrows():
            key = element.filename
            img = Image.open(io.BytesIO(self.config["store"][key]))
            img = img.convert("RGB")
            if self.image_processing is not None:
                img = self.image_processing(img)
            img = np.array(img)
            img = np.transpose(img, (2, 0, 1))
            images.append(np.array(img))
            times.append(timestamp)

        if images:
            images = np.array(images)
            channels = ["R", "G", "B"]
        else:
            images = np.empty((0, 0, 0, 0))
            # raise RuntimeError('No images in the selected range')

        data = xr.DataArray(
            images,
            coords={"time": times, "channels": channels},
            dims=["time", "channels", "x", "y"],
            name="Image",
        )
        data.attrs["format"] = "jpg"

        return data

    def construct_base64(self, filenames):
        images = []
        times = []
        for timestamp, element in filenames.iterrows():
            key = element.filename
            img_base64 = base64.b64encode(self.config["store"][key])
            images.append(img_base64)
            times.append(timestamp)

        images = np.array(images).reshape((-1, 1))
        data = xr.DataArray(
            images, coords={"time": times}, dims=["time", "base64"], name="Base64Image"
        )
        data.attrs["format"] = "jpg"

        return data
