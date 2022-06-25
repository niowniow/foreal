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

import re
from pathlib import Path
from warnings import warn

import pandas as pd
import xarray as xr

from foreal.config import get_setting, set_setting, setting_exists
from foreal.convenience import indexers_to_slices
from foreal.core import Node


class GSNDataSource(Node):
    def __init__(
        self,
        deployment=None,
        vsensor=None,
        position=None,
        device_id=None,
        fields=None,
        return_dataframe=False,
        indexers=None,
        **kwargs,
    ):
        # Make sure default arguments are non-mutable
        if indexers is None:
            indexers = {}

        super().__init__(
            deployment=deployment,
            position=position,
            device_id=device_id,
            vsensor=vsensor,
            fields=fields,
            return_dataframe=False,
            indexers=indexers,
            kwargs=kwargs,
        )

    def configure(self, requests=None):
        """Default configure for DataSource nodes
        Arguments:
            request {list} -- List of requests

        Returns:
            dict -- Original request or merged requests
        """
        request = super().configure(requests)  # merging request here

        new_request = {"requires_request": True}
        new_request.update(request)

        return new_request

    def forward(self, data=None, request=None):
        if request is None:
            return None

        # code from Samuel Weber
        #### 1 - DEFINE VSENSOR-DEPENDENT COLUMNS ####
        if get_setting("metadata_directory") is not None:
            metadata = get_setting(
                "metadata_directory"
            ) + "vsensor_metadata/{:s}_{:s}.csv".format(
                request["deployment"], request["vsensor"]
            )

            colnames = pd.read_csv(
                metadata,
                skiprows=0,
            )
            columns_old = colnames["colname_old"].values
            columns_new = colnames["colname_new"].values
            columns_unit = colnames["unit"].values

            if len(columns_old) != len(columns_new):
                warn(
                    "WARNING: Length of 'columns_old' ({:d}) is not equal length of 'columns_new' ({:d})".format(
                        len(columns_old), len(columns_new)
                    )
                )
            if len(columns_old) != len(columns_unit):
                warn(
                    "WARNING: Length of 'columns_old' ({:d}) is not equal length of 'columns_unit' ({:d})".format(
                        len(columns_old), len(columns_unit)
                    )
                )

            unit = dict(zip(columns_new, columns_unit))
        else:
            warn("WARNING: No meta data directory set for foreal")
            columns_old = []
            columns_new = None
            unit = None

        #### 2 - DEFINE CONDITIONS AND CREATE HTTP QUERY ####

        # Set server
        server = get_setting("permasense_server")

        # Create virtual_sensor
        virtual_sensor = request["deployment"] + "_" + request["vsensor"]
        fields = "All" if request["fields"] is None else request["fields"]

        # Create query and add time as well as position / device selection
        query = (
            "vs[1]={:s}"
            "&time_format=iso"
            "&timeline=generation_time"
            "&field[1]={:s}"
            "&from={:s}"
            "&to={:s}"
            "&c_vs[1]={:s}"
            "&c_join[1]=and"
        ).format(
            virtual_sensor,
            fields,
            pd.to_datetime(request["indexers"]["time"]["start"], utc=True).strftime(
                "%d/%m/%Y+%H:%M:%S"
            ),
            pd.to_datetime(request["indexers"]["time"]["stop"], utc=True).strftime(
                "%d/%m/%Y+%H:%M:%S"
            ),
            virtual_sensor,
        )

        if request["position"] is not None:
            query += (
                "&c_field[1]=position" "&c_min[1]={:02d}" "&c_max[1]={:02d}"
            ).format(
                int(request["position"]) - 1,
                request["position"],
            )
        if request["device_id"] is not None:
            query += (
                "&c_field[1]=device_id" "&c_min[1]={:02d}" "&c_max[1]={:02d}"
            ).format(
                int(request["device_id"]) - 1,
                request["device_id"],
            )

        # query extension for images
        if request["vsensor"] == "binary__mapped":
            query = (
                query
                + "&vs[2]={:s}&field[2]=relative_file&c_join[2]=and&c_vs[2]={:s}&c_field[2]=file_complete&c_min[2]=0&c_max[2]=1&vs[3]={:s}&field[3]=generation_time&c_join[3]=and&c_vs[3]={:s}&c_field[3]=file_size&c_min[3]=2000000&c_max[3]=%2Binf&download_format=csv".format(
                    virtual_sensor, virtual_sensor, virtual_sensor, virtual_sensor
                )
            )

        # Construct url:
        url = server + "multidata?" + query
        # print('The GSN http-query is:\n{:s}'.format(url))

        #### 3 - ACCESS DATA AND CREATE PANDAS DATAFRAME ####
        try:
            d = pd.read_csv(
                url, skiprows=2
            )  # skip header lines (first 2) for import: skiprows=2
        except pd.errors.EmptyDataError:
            print("Request contained no data:\n{:s}".format(url))
            return None

        # Remove '#' from first column name
        d.rename(columns={d.columns[0]: d.columns[0].replace("#", "")}, inplace=True)

        # Convert timestamps
        df = pd.DataFrame(columns=columns_new)
        df.time = pd.to_datetime(d.generation_time, utc=True)

        # if depo in ['mh25', 'mh27old', 'mh27new', 'mh30', 'jj22', 'jj25', 'mh-nodehealth']:
        # d = d.convert_objects(convert_numeric=True)  # TODO: rewrite to remove warning
        for k in list(df):
            df[k] = pd.to_numeric(df[k], errors="ignore")

        for i in range(len(columns_old)):
            column_old = columns_old[i].replace(
                "#", ""
            )  # Make sure that all column names are treated equally (meta-file contains # for the first column if no filtering is applied)
            if columns_new[i] != "time" and column_old in d.columns:
                setattr(df, columns_new[i], getattr(d, column_old))

        df = df.sort_values(by="time")

        # Remove columns with names 'delete'
        try:
            df.drop(["delete"], axis=1, inplace=True)
        except Exception as ex:
            warn("WARNING: Could not drop columns: {:s}".format(ex))

        # Remove columns with only 'null'
        # df = df.replace(r"null", np.nan, regex=True)
        isnull = df.isnull().all()
        [df.drop([col_name], axis=1, inplace=True) for col_name in df.columns[isnull]]

        # Sort DataFrame according to generation time and set as index
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
        df = df.sort_index(axis=1)

        # Convert to DataArray
        x = xr.DataArray(df, dims=["time", "name"], name="CSV")

        # Rewrite column name
        if unit is not None:
            try:
                unit_coords = []
                for name in x.coords["name"].values:
                    # name = re.sub(r"\[.*\]", "", name).lstrip().rstrip()
                    u = unit[str(name)]
                    u = re.findall(r"\[(.*)\]", u)[0]

                    # name_coords.append(name)
                    unit_coords.append(u)

                x = x.assign_coords({"unit": ("name", unit_coords)})
            except Exception as ex:
                warn("WARNING: Could not find a suitable unit; error: {:s}".format(ex))

        # Slice time and column names in DataArray according to indexers
        x = x.sel(indexers_to_slices(request["indexers"]))
        # x = x.loc[indexers_to_slices(request['indexers'])]

        if request.get("return_dataframe", False):
            return x.to_dataframe()
        else:
            return x
