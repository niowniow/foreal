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
from threading import Lock

import dask
import dask.array
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from benedict import benedict
from dask.utils import SerializableLock

import foreal
from foreal.convenience import dict_update, exclusive_indexing, indexers_to_slices
from foreal.core import Node
from foreal.core.graph import Node, NodeFailedException
from rechunker import rechunk

# from dask.distributed import wait


def convert_nestedfrozen(request):
    for key in request:
        if isinstance(request.get(key), dict):
            request[key] = dict(request[key])
            request[key] = convert_nestedfrozen(request[key])

        return request


def make_group_name(request, group_keys, sep="/"):
    vals = []
    sorted_group_keys = sorted(group_keys)

    # convert_nestedfrozen(request)
    # wrapping it in a benedict allows us to use dot syntax as group_key
    # to access keys in subdictionaries -> key.subkey
    benedict_request = benedict(request)
    for key in sorted_group_keys:
        if key not in benedict_request:
            raise RuntimeError(
                f"The group key {key} is not present in the given request: {benedict_request}"
            )
        val = benedict_request[key]
        if isinstance(val, list):
            val = sep.join(val)
        if not isinstance(val, str):
            raise RuntimeError(
                f"Values defined by group_keys must be str not {type(val)} with value {val}"
            )
        vals += [val]
    return sep.join(vals)


class DistributedWriter(Node):
    def __init__(
        self,
        store,
        dataset_start,
        dynamic_dim,
        step,
        synchronizer,
        group,
        i,
        region_start,
        region_stop,
        start,
        stop,
        chunk_start,
        chunk_stop,
        dask_key_name,
        data_prototype,
        bypass_storage=False,
    ):
        super().__init__()

        self.store = store
        self.dataset_start = dataset_start
        self.dynamic_dim = dynamic_dim
        self.step = step
        self.synchronizer = synchronizer
        self.group = group
        self.i = i
        self.region_start = region_start
        self.region_stop = region_stop
        self.start = start
        self.stop = stop
        self.chunk_start = chunk_start
        self.chunk_stop = chunk_stop
        self.parent_dask_key_name = dask_key_name
        self.dask_key_name = dask_key_name + "_distributed_writer"
        self.data_prototype = data_prototype
        self.bypass_storage = bypass_storage

    def optimize(self, class_nodes):
        pass

    def configure(self, requests=None):
        request = requests[0]
        new_request = {"requires_request": True}
        new_request.update(request)

        return new_request

    def __dask_tokenize__(self):
        return (
            DistributedWriter,
            self.store,
            self.group,
            self.region_start,
            self.region_stop,
            self.bypass_storage,
        )

    def forward(self, dataarray=None, request=None):
        # store to disk into a region and return dataarray
        if isinstance(dataarray, list):
            # TODO (niowniow): Rewrite foreal forward data arguments to not have
            #       list of one elements as return value
            dataarray = dataarray[0]

        # mark this instance as failed to not load it again
        availability = zarr.open(
            self.store,
            path=self.group + "/" + self.dynamic_dim + "_chunks",
            mode="r+",
            synchronizer=self.synchronizer,
        )

        if isinstance(dataarray, NodeFailedException):
            # TODO: make verbose mode to give indication if it failed
            availability[self.chunk_start : self.chunk_stop] = -1

            return None  # FIXME
            return dataarray

        # let's check if it is already computed what we want
        # this could happen this distributedwriter instance had to be recomputed
        # because it's worker ended or a different process computed the chunk beofre this writer
        # was scheduled.
        all_computed = [
            availability[i] == 1 for i in range(self.chunk_start, self.chunk_stop)
        ]
        if all(all_computed):
            return None

        if self.bypass_storage:
            return dataarray

        # def correct_frame_extracted(data):
        #     # FIXME: self.dynamic_dim non-time
        #     start_cond = data[self.dynamic_dim] >= pd.to_datetime(self.start)
        #     stop_cond = data[self.dynamic_dim] < pd.to_datetime(self.stop)
        #     return bool(start_cond.all() and stop_cond.all())

        # if not correct_frame_extracted(dataarray):
        #            raise RuntimeError('')
        # create empty dataarray with a regularly sampled grid and
        # integrate input dataarray to it (i.e. adding nans if samples are missing)
        if dataarray.name is None:
            dataarray.name = "data"

        template_length = self.region_stop - self.region_start
        # if len(dataarray[self.dynamic_dim].values) == 0:
        template_coordinate = self.start + np.arange(template_length) * self.step
        # else:
        #     template_coordinate = self.start + (dataarray[self.dynamic_dim].values[0] - self.start) % self.step + np.arange(template_length) * self.step
        shape = list(self.data_prototype.shape)
        index_dynamic_dim = self.data_prototype.dims.index(self.dynamic_dim)
        shape[index_dynamic_dim] = template_length
        template_shape = tuple(shape)
        template = np.full(template_shape, np.nan)
        # print(self.data_prototype)
        coords = {
            dim: self.data_prototype.coords[dim]
            for dim in self.data_prototype.dims
            if dim != self.dynamic_dim
        }
        coords[self.dynamic_dim] = template_coordinate

        template_da = xr.DataArray(
            template,
            dims=dataarray.dims,
            coords=coords,
            name=dataarray.name,
        )

        merged_dataset = xr.merge([template_da, dataarray], join="left")
        dataarray = merged_dataset[dataarray.name]
        # merged_dataset.attrs = dataarray.attrs

        if dataarray.shape != template_da.shape:
            raise RuntimeError(
                f"Data error: Shapes do not match for DataArray with shape {dataarray.shape}. Expected {template_da.shape}"
            )

        # to avoid that xarray loads the whole dynamic_dim coordinate into memory
        # we create a non-dimensional coordinate of our dimension coordinate
        # Moreover, to avoid that xarray converts the whole coordinate
        # to datetime automatically
        # we do the conversion manually only for the region we want to store
        # convert to nanoseconds since dataset_start
        dynamic_dim_values = dataarray[self.dynamic_dim].values
        if pd.api.types.is_datetime64_any_dtype(dynamic_dim_values) or isinstance(
            self.start, pd.Timestamp
        ):
            dynamic_dim_values = pd.to_datetime(dynamic_dim_values).tz_localize(None)
        #        dynamic_dim_nondim = dynamic_dim_values - self.dataset_start.tz_localize(None)
        dynamic_dim_nondim = dynamic_dim_values - self.dataset_start
        if pd.api.types.is_datetime64_any_dtype(dynamic_dim_values) or isinstance(
            self.start, pd.Timestamp
        ):
            dynamic_dim_nondim = dynamic_dim_nondim / pd.to_timedelta(1, "ns")

        dataarray = dataarray.assign_coords(
            {self.dynamic_dim + "_nondim": (self.dynamic_dim, dynamic_dim_nondim)}
        )

        adjusted_region_stop = self.region_stop - (
            (self.region_stop - self.region_start) - dataarray.sizes[self.dynamic_dim]
        )
        del dataarray[self.dynamic_dim]
        dataset = dataarray.to_dataset(name=self.parent_dask_key_name)
        dataset = dataset.assign({self.dynamic_dim + "_chunks": [1]})

        # build our region selector to access the zarr storage
        region_sel = {
            dim: slice(dataset.sizes[dim])
            for dim in dataset.dims
            if (dim != self.dynamic_dim and dim != self.dynamic_dim + "_chunks")
        }
        region_sel[self.dynamic_dim] = slice(self.region_start, adjusted_region_stop)
        region_sel[self.dynamic_dim + "_chunks"] = slice(
            self.chunk_start, self.chunk_stop
        )

        # FIXME: a hack storing the data manually
        x = zarr.open(
            self.store,
            path=self.group + "/" + self.parent_dask_key_name,
            mode="a",
            synchronizer=self.synchronizer,
        )
        da = dataset[self.parent_dask_key_name]
        slicer = [
            slice(None)
            if da.dims.index(self.dynamic_dim) != i
            else np.arange(
                region_sel[self.dynamic_dim].start, region_sel[self.dynamic_dim].stop
            ).tolist()
            for i in range(len(da.shape))
        ]
        x.oindex[tuple(slicer)] = dataset[self.parent_dask_key_name].values

        key = self.dynamic_dim + "_chunks"
        x = zarr.open(
            self.store,
            path=self.group + "/" + key,
            mode="a",
            synchronizer=self.synchronizer,
        )
        x[region_sel[key].start : region_sel[key].stop] = dataset[key].values

        key = self.dynamic_dim + "_nondim"
        x = zarr.open(
            self.store,
            path=self.group + "/" + key,
            mode="a",
            synchronizer=self.synchronizer,
        )
        x[
            region_sel[self.dynamic_dim].start : region_sel[self.dynamic_dim].stop
        ] = dataset[key].values
        # FIXME: end of hack

        del dataarray

        return


class Persister(Node):
    def __init__(
        self,
        store,
        dataset_scope=None,
        prototype_request=None,
        processing_graph=None,
        dynamic_dim="time",
        group_keys=None,
        synchronizer=None,
        chunk_cache=None,
        chunk_size=None,
        raw=False,
        bypass=False,
        bypass_storage=False,
        auto_initialize=False,
    ):
        """
        Note: The prototype_request should contain group keys as globals
        i.e. {'config':{'global':{'group_key_name':'value'}}}
        otherwise it might lead to errors, since in any request the group key will be
        overwritten if they are in the `types` or `keys` config section of the
        prototype_request.
        """
        super().__init__(raw=raw, bypass=bypass, bypass_storage=bypass_storage)
        if isinstance(store, str) or isinstance(store, Path):
            store = zarr.DirectoryStore(store)
        self.store = store
        self.dataset_scope = dataset_scope
        self.raw = {}
        self.ds = {}
        self.availability = {}
        self.info = {}
        self._mutex = SerializableLock()
        # # FIXME: make it work for non-time coordinates
        # if dynamic_dim != "time":
        #     raise NotImplementedError(
        #         f"Persister is only implemented for dynamic_dim='time' which must be a datetime coordinate"
        #     )
        self.dynamic_dim = dynamic_dim
        self.group_keys = group_keys if group_keys is not None else []
        self.synchronizer = synchronizer
        self.prototype_request = prototype_request
        self.chunk_size = chunk_size
        self.ds_map = {}
        self.attrs = {}
        if isinstance(chunk_cache, int):
            self.chunk_cache = zarr.LRUChunkCache(max_size=chunk_cache)
        else:
            self.chunk_cache = chunk_cache

        # Initialize empty class variables
        self.processing_graph = processing_graph
        self.data_prototypes = {}
        self.dynamic_dim_is_datetime = False
        self.dataset_start = None
        self.dataset_end = None
        self.shape = None
        self.chunk_shape = None
        self.step = None
        self.auto_initialize = auto_initialize

    def __call__(self, *args, **kwargs):
        super_return = super().__call__(*args, **kwargs)
        if self.auto_initialize:
            if "data" not in kwargs:
                processing_graph = args[0]
            else:
                processing_graph = kwargs.get("data", None)
            self.initialize(processing_graph=processing_graph)

        return super_return

    def make_group_name(self, request, sep="/"):
        return make_group_name(request, self.group_keys, sep=sep)

    def initialize(
        self,
        processing_graph=None,
        request=None,
        dataset_scope=None,
        is_prototype=False,
        sampling_rate=None,
        step=None,
        mode="w-",
        threaded=False,
    ):
        if processing_graph is not None:
            self.processing_graph = processing_graph

        if dataset_scope is not None:
            self.dataset_scope = dataset_scope

        if self.processing_graph is None:
            raise RuntimeError(
                "No processing_graph provided. "
                "Add it during instantiation or during the call to initialize"
            )

        if request is not None:
            request = self.merge_config(request)
            if is_prototype:
                prototype_request = request
                if self.prototype_request is None:
                    self.prototype_request = deepcopy(dict(request))
                for gkey in self.group_keys:
                    tmp_dict = benedict()
                    tmp_dict[gkey] = benedict(request["self"])[gkey]
                    dict_update(prototype_request, tmp_dict)
                    # dict_update(prototype_request,{gkey:deepcopy(request["self"][gkey])})
            else:
                prototype_request = deepcopy(dict(self.prototype_request))
                for gkey in self.group_keys:
                    tmp_dict = benedict()
                    tmp_dict[gkey] = benedict(request["self"])[gkey]
                    dict_update(prototype_request, tmp_dict)
                    # dict_update(prototype_request,{'config':{'global':{gkey:deepcopy(request["self"][gkey])}}})
                    # dict_update(prototype_request,{gkey:deepcopy(request["self"][gkey])})
        else:
            prototype_request = self.merge_config(self.prototype_request)
            for gkey in self.group_keys:
                tmp_dict = benedict()
                tmp_dict[gkey] = benedict(prototype_request["self"])[gkey]
                dict_update(prototype_request, tmp_dict)
                # dict_update(prototype_request,{gkey:deepcopy(prototype_request["self"][gkey])})

        group = self.make_group_name(prototype_request)

        dynamic_dim = self.dynamic_dim

        # # check if it's already available
        try:
            attrs = zarr.open_group(self.store, path=group, mode="r+").attrs

            # check if all required data
            self.step = attrs["step"]
            self.dynamic_dim = attrs["dynamic_dim"]
            self.dynamic_dim_is_datetime = attrs["dynamic_dim_is_datetime"]
            self.dataset_start = attrs["dataset_start"]
            self.dataset_end = attrs["dataset_end"]
            if self.dynamic_dim_is_datetime:
                self.step = pd.to_timedelta(self.step)
                self.dataset_start = pd.to_datetime(self.dataset_start).tz_localize(
                    None
                )
                self.dataset_end = pd.to_datetime(self.dataset_end).tz_localize(None)

            self.shape = attrs["shape"]
            self.chunk_size = attrs["chunk_size"]
            self.chunk_shape = attrs["chunk_shape"]

            drop_variables = None
            if self.isfinalized(group=group):
                rechunk_variables = [
                    self.dynamic_dim + "_chunks",
                    self.dynamic_dim + "_nondim",
                ]
                # try to open reschuelde
                drop_variables = []
                for var in rechunk_variables:
                    dim_group = zarr.open(
                        self.store,
                        path=group + "/" + var,
                        mode="r",
                        chunk_cache=self.chunk_cache,
                    )
                    print(dim_group.shape)
                    print(self.store)
                    print(var)
                    if not dim_group.attrs.get("rechunked", False):
                        # the group not already the rechunked version. do it
                        # in_memory_x = np.array(dim_group) # this reads the whole zarr array into memory
                        # new_array = zarr.array(in_memory_x, chunks=in_memory_x.shape, store=self.store, path=group + "/" + var + "_finalized_tmp")
                        shape = ("auto",)
                        target_chunks = dask.array.core.auto_chunks(
                            shape, dim_group.shape, limit=None, dtype=dim_group.dtype
                        )
                        array_plan = rechunk(
                            dim_group,
                            target_chunks,
                            "1000MB",
                            target_store=self.store,
                            target_options={
                                "path": group + "/" + var + "_finalized_tmp"
                            },
                            temp_store=self.store,
                            temp_options={"path": group + "/" + var + "_tmp"},
                        )
                        tmp_array = array_plan.execute()
                        tmp_array.attrs["rechunked"] = True
                        zarr_group = zarr.open(
                            self.store,
                            path=group,
                            mode="a",
                            chunk_cache=self.chunk_cache,
                        )

                        zarr_group.move(var, var + "_nchunks")
                        zarr_group.move(var + "_finalized_tmp", var)

                        drop_variables += [var + "_nchunks"]

            ds = xr.open_zarr(
                self.store,
                group=group,
                chunk_cache=self.chunk_cache,
                chunks="auto",
                drop_variables=drop_variables,
            )

            da = ds[self.dask_key_name]
            self.data_prototypes[group] = da
            # TODO:
            # 1. create a data_prototype from stored data
            # 2. Return data_prototype
            self.attrs[group] = attrs

            return da

        except (zarr.errors.GroupNotFoundError, zarr.errors.PathNotFoundError):
            # if any key does not exist or the whole store is empty
            # we need to initialize all again.
            print("Initializing persister")
            pass
        except KeyError as e:
            print("Initializing persister")
            pass
        except Exception as e:
            raise RuntimeError(e)

        if isinstance(self.processing_graph, xr.DataArray):
            data_prototype = self.processing_graph
        else:
            if not threaded:
                with dask.config.set(scheduler="single-threaded"):
                    x_data_conf = foreal.core.configuration(
                        self.processing_graph, prototype_request, optimize_graph=False
                    )
                    x_data_computed = dask.compute(x_data_conf)[0]
            else:
                x_data_conf = foreal.core.configuration(
                    self.processing_graph, prototype_request, optimize_graph=False
                )

                x_data_computed = dask.compute(x_data_conf)[0]
            data_prototype = x_data_computed
        self.data_prototypes[group] = data_prototype
        # TODO: if data_prototype is a tuple, iterate through and save each element in
        #       in a separate group!

        if isinstance(data_prototype, NodeFailedException):
            raise RuntimeError(
                f"For initializing persist: Please provide a request returning valid data. Failed request: {prototype_request} with error {str(data_prototype)}"
            )

        if self.step is not None:
            step = self.step
        if step is None:
            # TODO: remove sampling rate
            if sampling_rate is None:
                if len(data_prototype[dynamic_dim]) < 2:
                    raise RuntimeError(
                        f"Please provide a prototype request which yields a data sample with length > 1 on dimension {dynamic_dim} or provide the step argument"
                    )
                # must be sorted
                min_val = data_prototype[dynamic_dim][0]
                next_val = data_prototype[dynamic_dim][1]

                # if isinstance(min_val.values,np.ndarray):
                #     min_val = min_val.values[0]
                # if isinstance(next_val.values,np.ndarray):
                #     next_val = next_val.values[0]
                step = next_val - min_val
                # max_val = data_prototype['time'][-1]
                step = step.values
            else:
                step = pd.to_timedelta(1 / sampling_rate, "s")
        # self.prototype_request = prototype_request

        self.step = step
        self.dynamic_dim = dynamic_dim
        # self.dynamic_dim_type = data_prototype[dynamic_dim].dtype
        self.dynamic_dim_is_datetime = pd.api.types.is_datetime64_any_dtype(
            data_prototype[dynamic_dim].dtype
        )

        if self.dataset_scope is None:
            raise RuntimeError(
                "No dataset_scope provided."
                "Add it during instantiation or during the call to initialize"
            )

        dataset_start = self.dataset_scope[dynamic_dim]["start"]
        dataset_end = self.dataset_scope[dynamic_dim]["stop"]
        if self.dynamic_dim_is_datetime:
            self.step = pd.to_timedelta(self.step)
            dataset_start = pd.to_datetime(dataset_start).tz_localize(None)
            dataset_end = pd.to_datetime(dataset_end).tz_localize(None)

        self.dataset_start = dataset_start
        self.dataset_end = dataset_end
        # chunk_size = data_prototype.sizes[dynamic_dim]

        num_elements = (dataset_end - dataset_start) // step

        # compute the new dataarray shape for the whole persist period
        shape = list(data_prototype.shape)
        index_dynamic_dim = data_prototype.dims.index(dynamic_dim)
        # num_chunks = shape[index_dynamic_dim]
        shape[index_dynamic_dim] = int(num_elements)
        self.shape = tuple(shape)

        if self.chunk_size is not None:
            shape[index_dynamic_dim] = self.chunk_size
            chunk = tuple(shape)
            self.chunk_shape = dict(zip(data_prototype.dims, chunk))
            dummies = dask.array.zeros(self.shape, chunks=chunk)
        else:
            # dummies = dask.array.zeros(tuple(shape), chunks=data_prototype.shape)
            dummies = dask.array.zeros(
                self.shape, dtype=data_prototype.dtype
            )  # determining the chunksize automatically

            self.chunk_size = dummies.chunksize[index_dynamic_dim]
            self.chunk_shape = dict(zip(data_prototype.dims, dummies.chunksize))
        chunk_size = self.chunk_size

        coord_dynamic = dask.array.zeros(
            (int(num_elements),), chunks=chunk_size, dtype=np.float64
        )
        chunks_dynamic = dask.array.zeros(
            (int(np.ceil(num_elements / chunk_size)),), dtype=np.int8
        )

        # coord_dynamic = dask.array.zeros((int(num_elements),), dtype=np.float64)

        # # chunk_size = coord_dynamic.info
        # chunks_dynamic = dask.array.zeros((int(np.ceil(num_elements/chunk_size)),), dtype=np.bool)

        # coord_dynamic = dask.array.zeros((int(num_elements),), chunks=data_prototype.shape[-1:], dtype=data_prototype['time'].dtype)
        # coord_dynamic = dask.array.zeros((int(num_elements),), chunks=data_prototype.shape[-1:], dtype=data_prototype['time'].dtype)

        coords = {
            dim: data_prototype.coords[dim]
            for dim in data_prototype.dims
            if dim != dynamic_dim
        }
        coords[dynamic_dim + "_nondim"] = (dynamic_dim, coord_dynamic)

        ds = xr.DataArray(dummies, dims=data_prototype.dims, coords=coords).to_dataset(
            name=self.dask_key_name
        )
        ds = ds.assign({dynamic_dim + "_chunks": chunks_dynamic})

        try:
            x = ds.to_zarr(
                self.store,
                group=group,
                mode=mode,
                compute=False,
                consolidated=True,
                encoding={dynamic_dim + "_chunks": {"chunks": 1}},
            )

            # x = zarr.open_array(
            #     self.store,
            #     path=group + "/" + dynamic_dim + "_chunks",
            #     mode="w",
            #     synchronizer=self.synchronizer,
            #     shape=chunks_dynamic.shape,
            #     dtype=chunks_dynamic.dtype,
            #     chunks=1,
            # )
        except zarr.errors.ContainsGroupError:
            print("Using persister dataset already present at", self.store.path)
            pass

        attrs = zarr.open_group(self.store, path=group, mode="r+").attrs
        attrs["dynamic_dim"] = self.dynamic_dim
        attrs["dynamic_dim_is_datetime"] = self.dynamic_dim_is_datetime
        step = self.step
        dataset_start = self.dataset_start
        dataset_end = self.dataset_end
        if self.dynamic_dim_is_datetime:
            attrs["step"] = pd.to_timedelta(step).isoformat()
            attrs["dataset_start"] = pd.to_datetime(dataset_start).isoformat()
            attrs["dataset_end"] = pd.to_datetime(dataset_end).isoformat()
        else:
            attrs["step"] = step
            attrs["dataset_start"] = dataset_start
            attrs["dataset_end"] = dataset_end

        attrs["shape"] = self.shape
        attrs["chunk_size"] = self.chunk_size
        attrs["chunk_shape"] = self.chunk_shape
        self.attrs[group] = attrs

        return data_prototype

    # def cache_me_if_you_can(self, x, group=""):
    #     if self.isfinalized(group=group):
    #         return np.array(x) # this reads the whole zarr array into memory

    #     return x

    def cache_me_if_you_can(self, x, group=""):
        if self.isfinalized(group=group):
            # try:
            #     # we try to read from the rechunked version of the group
            #     x_final = zarr.open(
            #             self.store,
            #             path=group + "/" + self.dynamic_dim + "_chunks_finalized",
            #             mode="r",
            #             chunk_cache=self.chunk_cache,
            #         )
            #     return np.array(x_final)
            # except:
            #     in_memory_x = np.array(x) # this reads the whole zarr array into memory
            #     new_x = zarr.array(in_memory_x, chunks=in_memory_x.shape, store=self.store, path=group + "/" + self.dynamic_dim + "_chunks_finalized")
            #     return in_memory_x
            in_memory_x = np.array(x)  # this reads the whole zarr array into memory
            return in_memory_x

        return x

    def get_availability(self, group=None, request=None):
        # Make sure default arguments are non-mutable
        if group is None:
            group = []

        try:
            # Open zarr store directly (bypassing xarray). It's faster!
            # TODO: we could possibly also cache the  xarray.open_zarr() call
            #       and access the subgroup `self.dynamic_dim + "_chunks"`
            #       might be as fast and a cleaner total solution
            if group not in self.availability:
                self.availability[group] = self.cache_me_if_you_can(
                    zarr.open(
                        self.store,
                        path=group + "/" + self.dynamic_dim + "_chunks",
                        mode="r",
                        chunk_cache=self.chunk_cache,
                    ),
                    group=group,
                )
            if group not in self.data_prototypes:
                raise RuntimeError("Persister is not initialized")
        except Exception as e:
            print(f"Info: {e}. Initializing the store and trying again")
            self.initialize(request=request)
            # Open zarr store directly (bypassing xarray). It's faster!

            self.availability[group] = self.cache_me_if_you_can(
                zarr.open(
                    self.store,
                    path=group + "/" + self.dynamic_dim + "_chunks",
                    mode="r",
                ),
                group=group,
            )
        return self.availability[group]

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
        if request["self"]["bypass"]:
            return new_request

        group = self.make_group_name(request["self"])
        availability = self.get_availability(group, request)
        if group not in self.info:
            info = zarr.open(
                self.store,
                path=group + "/" + self.dynamic_dim + "_nondim",
                mode="r",
            )

            self.info[group] = info.chunks[0]
        chunk_size = self.info[group]

        start_value = request["self"]["indexers"][self.dynamic_dim]["start"]
        if self.dynamic_dim_is_datetime:
            start_value = foreal.to_datetime(start_value).tz_localize(None)

        start_index_from_request = (
            (start_value - self.dataset_start) / self.step / chunk_size
        )
        start_index_from_request = int(np.floor(start_index_from_request))
        start_index_from_request = np.maximum(start_index_from_request, 0)

        stop_value = request["self"]["indexers"][self.dynamic_dim]["stop"]
        if self.dynamic_dim_is_datetime:
            stop_value = foreal.to_datetime(stop_value)

        stop_index_from_request = (
            (stop_value - self.dataset_start) / self.step / chunk_size
        )
        stop_index_from_request = int(np.ceil(stop_index_from_request))

        max_index = (self.dataset_end - self.dataset_start) / self.step / chunk_size

        def bound_range(x):
            x = np.maximum(x, 0)
            x = np.minimum(x, max_index)
            return int(x)

        start_index_from_request = bound_range(start_index_from_request)
        stop_index_from_request = bound_range(stop_index_from_request)

        availability = availability[start_index_from_request:stop_index_from_request]

        # ds = xr.open_zarr(self.store)
        # availability = ds[self.dynamic_dim+'_chunks'].isel({self.dynamic_dim+'_chunks':slice(start_index_from_request,stop_index_from_request+1)})

        starts = (
            np.arange(start_index_from_request, stop_index_from_request)
            * self.step
            * chunk_size
            + self.dataset_start
        )
        stops = starts + self.step * chunk_size

        insertions = []
        clone_dependencies = []
        for i in range(len(starts)):
            if i >= len(availability):
                continue

            if availability[i] == 0 or request["self"].get("bypass_storage", False):
                sub_request = deepcopy(dict(self.prototype_request))
                # dict_update(sub_request,request)
                for gkey in self.group_keys:
                    # sub_request[gkey] = deepcopy(request["self"][gkey])
                    # dict_update(sub_request,{'config':{'global':{gkey:deepcopy(request["self"][gkey])}}})
                    tmp_dict = benedict()
                    tmp_dict[gkey] = benedict(request["self"])[gkey]
                    dict_update(sub_request, tmp_dict)
                # start = pd.to_datetime(starts[i])
                start = starts[i]
                # stop = pd.to_datetime(stop_times[i])
                stop = stops[i]

                if self.dynamic_dim_is_datetime:
                    start = pd.to_datetime(start).tz_localize(None)
                    stop = pd.to_datetime(stop).tz_localize(None)

                # region_start = int((start - self.dataset_start) / self.step)
                region_start = int(np.ceil((start - self.dataset_start) / self.step))
                region_stop = int(np.floor((stop - self.dataset_start) / self.step))

                # region_stop = int((stop - self.dataset_start) / self.step)
                sub_request["indexers"][self.dynamic_dim]["start"] = start
                sub_request["indexers"][self.dynamic_dim]["stop"] = stop

                if self.dynamic_dim_is_datetime:
                    sub_request["indexers"][self.dynamic_dim][
                        "start"
                    ] = start.isoformat()
                    sub_request["indexers"][self.dynamic_dim]["stop"] = stop.isoformat()

                chunk_start = start_index_from_request + i
                chunk_stop = chunk_start + 1

                insertions += [
                    DistributedWriter(
                        store=self.store,
                        dataset_start=self.dataset_start,
                        dynamic_dim=self.dynamic_dim,
                        step=self.step,
                        synchronizer=self.synchronizer,
                        group=group,
                        i=i,
                        region_start=region_start,
                        region_stop=region_stop,
                        start=start,
                        stop=stop,
                        chunk_start=chunk_start,
                        chunk_stop=chunk_stop,
                        dask_key_name=self.dask_key_name,
                        data_prototype=self.data_prototypes[group],
                        bypass_storage=request["self"].get("bypass_storage", False),
                    ).forward
                ]
                clone_dependencies += [sub_request]

        if clone_dependencies:
            new_request["clone_dependencies"] = clone_dependencies
            new_request["insert_predecessor"] = insertions
        else:
            new_request["remove_dependencies"] = True
        new_request["requires_request"] = True

        return new_request

    def isfinalized(self, set_value=None, group=""):
        try:
            if group not in self.attrs or self.attrs[group] is None:
                self.attrs[group] = zarr.open_group(
                    self.store, path=group, mode="r+"
                ).attrs
        except zarr.errors.GroupNotFoundError:
            return False

        if set_value is not None:
            self.attrs[group]["finalized"] = set_value

        if "finalized" in self.attrs[group]:
            return self.attrs[group]["finalized"]

        return False

    def get_region_bounds(self, request):
        if self.dynamic_dim_is_datetime:
            indexers = deepcopy(dict(request["indexers"]))
            indexers[self.dynamic_dim]["start"] = pd.to_datetime(
                indexers[self.dynamic_dim]["start"]
            ).tz_localize(None)
            indexers[self.dynamic_dim]["stop"] = pd.to_datetime(
                indexers[self.dynamic_dim]["stop"]
            ).tz_localize(None)
        else:
            indexers = request["indexers"]

        start_value = indexers[self.dynamic_dim]["start"]

        region_start = int(np.ceil((start_value - self.dataset_start) / self.step))

        closed_right = False  # TODO: implement intervals correctly
        stop_value = indexers[self.dynamic_dim]["stop"]

        region_stop = int(np.floor((stop_value - self.dataset_start) / self.step))

        if closed_right:
            region_stop += 1

        return region_start, region_stop

    def isavailable(self, request, merge_config=False):
        if merge_config:
            request = self.merge_config(request)["self"]
        region_start, region_stop = self.get_region_bounds(request)
        group = self.make_group_name(request)

        availability_start = int(np.floor(region_start / self.chunk_size))
        availability_stop = int(np.ceil(region_stop / self.chunk_size))
        availability = self.get_availability(group, request)
        requested_availability = availability[availability_start:availability_stop]
        return requested_availability

    def open_group(group):
        self.ds[group] = xr.open_zarr(
            self.store,
            group=group,
            chunk_cache=self.chunk_cache,
            chunks=None,
        )

    def forward(self, data, request):
        if request["self"]["bypass"]:
            return data

        r = request["self"]
        if self.dynamic_dim_is_datetime:
            indexers = deepcopy(dict(r["indexers"]))
            indexers[self.dynamic_dim]["start"] = pd.to_datetime(
                indexers[self.dynamic_dim]["start"]
            )
            indexers[self.dynamic_dim]["stop"] = pd.to_datetime(
                indexers[self.dynamic_dim]["stop"]
            )
        else:
            indexers = r["indexers"]

        slices = indexers_to_slices(indexers)

        if r.get("bypass_storage", False):
            data = [
                d
                for d in data
                if not isinstance(d, NodeFailedException) and d is not None
            ]
            data = xr.concat(data, self.dynamic_dim)
            data = data.sel(slices)
            data = exclusive_indexing(data, indexers)
            return data

        if self.dynamic_dim in slices:
            del slices[self.dynamic_dim]

        # FIXME: Make function pure... Remove calls to 'global state' i.e. to self
        #        group = make_group_name(r, self.group_keys)
        #
        #        try:
        #            # TODO: we set chunks=None to avoid rechunking. Maybe in other situations
        #            #       it's a wrong default?
        #            ds = xr.open_zarr(self.store, group=group, chunk_cache=self.chunk_cache, chunks=None)
        #            ds = xr.open_zarr(self.store, group=group, chunk_cache=self.chunk_cache,chunks=self.chunk_shape)
        #        except Exception as e:
        #            print(e)
        #            print("initializing the store and trying again")
        #            self.initialize(request=r)
        #            ds = xr.open_zarr(self.store, group=group, chunk_cache=self.chunk_cache, chunks=None)

        # indexer_dynamic_dim = r["indexers"][self.dynamic_dim]
        # if not isinstance(indexer_dynamic_dim,dict):
        #     indexer_dynamic_dim  = {'start':}

        # start_value = r["indexers"][self.dynamic_dim]["start"]
        # if self.dynamic_dim_is_datetime:
        #     start_value = foreal.to_datetime(start_value).tz_localize(None)

        # region_start = int(
        #     np.ceil(
        #         (

        #             start_value - self.dataset_start
        #         )
        #         / self.step
        #     )
        # )

        # closed_right = False  # TODO: implement intervals correctly
        # stop_value = r["indexers"][self.dynamic_dim]["stop"]
        # if self.dynamic_dim_is_datetime:
        #     stop_value = foreal.to_datetime(stop_value).tz_localize(None)

        # region_stop = int(
        #         np.floor(
        #             (
        #                 stop_value
        #                 - self.dataset_start
        #             )
        #             / self.step
        #         )
        #     )

        # if closed_right:
        #     region_stop += 1

        # availability_start = int(np.floor(region_start/self.chunk_size))
        # availability_stop = int(np.ceil(region_stop/self.chunk_size))
        # availability = self.get_availability(group, request)
        # requested_availability = availability[availability_start:availability_stop]

        group = self.make_group_name(r)
        region_start, region_stop = self.get_region_bounds(r)
        requested_availability = self.isavailable(r)

        if (requested_availability == -1).all():
            raise NodeFailedException(
                f"No valid data in requested range available: {r}"
            )
        if (requested_availability == 0).any():
            raise NodeFailedException(
                "Not all required segments were preloaded."
                " Use `foreal.compute` with your task graph."
            )

        if r.get("raw", False):
            if group not in self.raw:
                self.raw[group] = zarr.open(
                    self.store,
                    path=group + "/" + self.dask_key_name,
                    mode="r",
                    chunk_cache=self.chunk_cache,
                )

            #            indexers = r['indexers']
            #            slices = indexers_to_slices(indexers)
            #            slicer = []
            #            for i in range(len(self.shape)):
            #                if self.data_prototypes[group].dims.index(self.dynamic_dim) == i:
            #                    slicer += [slice(region_start,region_stop)]
            #
            #                elif self.data_prototypes[group].dims[i] in slices:
            #                    dim = self.data_prototypes[group].dims[i]
            #                    slicer+= [[x in slices[dim] for x in self.data_prototypes[group][dim]]]
            #                    slicer+= [slices[self.data_prototypes[group].dims[i]]]
            #                else:
            #                    slicer += [slice(None)]
            #
            dynamic_dim_index = self.data_prototypes[group].dims.index(self.dynamic_dim)
            slice_multi_dim = any(
                [
                    dim in self.data_prototypes[group].dims
                    for dim in indexers
                    if dim != self.dynamic_dim
                ]
            )

            if slice_multi_dim:
                idxes = []
                for dim in self.data_prototypes[group].dims:
                    if dim == self.dynamic_dim:
                        # idx = np.arange(region_start, region_stop).tolist()
                        idx = slice(region_start, region_stop)
                    else:
                        if dim in indexers:
                            idx = (
                                self.data_prototypes[group]
                                .get_index(dim)
                                .get_indexer(indexers[dim])
                                .tolist()
                            )
                        else:
                            # idx = np.arange(len(self.data_prototypes[group][dim])).tolist()
                            idx = slice(None)
                    idxes += [idx]
                return_value = self.raw[group].oindex[tuple(idxes)]

            elif dynamic_dim_index == 0:
                return_value = self.raw[group][region_start:region_stop]
            elif dynamic_dim_index == self.data_prototypes[group].ndim - 1:
                return_value = self.raw[group][..., region_start:region_stop]
            else:
                slicer = [
                    slice(None)
                    if dynamic_dim_index != i
                    else np.arange(region_start, region_stop).tolist()
                    for i in range(len(self.shape))
                ]
                #            slicer = [slice(None) if da.dims.index(self.dynamic_dim) != i else np.arange(region_sel[self.dynamic_dim].start,region_sel[self.dynamic_dim].stop).tolist() for i in range(len(self.shape))]
                return_value = self.raw[group].oindex[tuple(slicer)]

            return return_value

        try:
            if group not in self.ds:
                with self._mutex:
                    if group not in self.ds:
                        self.ds[group] = xr.open_zarr(
                            self.store,
                            group=group,
                            chunk_cache=self.chunk_cache,
                            chunks=None,
                        )
        except Exception as e:
            print(e)
            print("initializing the store and trying again")
            self.initialize(request=r)
            if group not in self.ds:
                self.ds[group] = xr.open_zarr(
                    self.store, group=group, chunk_cache=self.chunk_cache, chunks=None
                )

        ds = self.ds[group]

        section = ds.isel({self.dynamic_dim: slice(region_start, region_stop)})

        dynamic_dim = self.dynamic_dim
        dynamic_dim_values = section[dynamic_dim + "_nondim"].values
        if self.dynamic_dim_is_datetime:
            dynamic_dim_values = pd.to_timedelta(dynamic_dim_values, "ns")

        section[dynamic_dim] = dynamic_dim_values + self.dataset_start.tz_localize(None)

        del section[dynamic_dim + "_nondim"]
        del section[dynamic_dim + "_chunks"]

        # get data array from dataset
        section = section[self.dask_key_name]
        section = section.sel(slices)
        section = exclusive_indexing(section, indexers)
        # print('d',r,section)
        # print(self.dynamic_dim_is_datetime)

        return section


class ImmutablePersist(Node):
    def __init__(
        self,
        store,
        prototype_request,
        group_keys=None,
        chunk_cache=None,
    ):
        super().__init__()
        if isinstance(store, str) or isinstance(store, Path):
            store = zarr.DirectoryStore(store)
        self.store = store
        self.ds = {}
        self._mutex = SerializableLock()

        self.group_keys = group_keys if group_keys is not None else []
        self.prototype_request = prototype_request

        if isinstance(chunk_cache, int):
            self.chunk_cache = zarr.LRUChunkCache(max_size=chunk_cache)
        else:
            self.chunk_cache = chunk_cache

        # Initialize empty class variables
        self.processing_graph = None

    def initialize(
        self,
        processing_graph=None,
        request=None,
        is_prototype=False,
        mode="w-",
    ):
        if processing_graph is None:
            try:
                processing_graph = self.processing_graph
            except Exception as ex:
                raise RuntimeError(
                    "Please call persister.initialize once with a task graph before using persister."
                )
        else:
            self.processing_graph = processing_graph

        if request is not None:
            if is_prototype:
                prototype_request = request
            else:
                prototype_request = deepcopy(dict(self.prototype_request))
                merged_request = self.merge_config(request)
                for gkey in self.group_keys:
                    tmp_dict = benedict()
                    tmp_dict[gkey] = benedict(merged_request["self"])[gkey]
                    dict_update(prototype_request, tmp_dict)
                    # dict_update(prototype_request,{'config':{'global':{gkey:deepcopy(merged_request["self"][gkey])}}})
                    # dict_update(prototype_request,{gkey:deepcopy(merged_request["self"][gkey])})
                    # prototype_request[gkey] = deepcopy(request[gkey])
        else:
            prototype_request = self.prototype_request
        group = self.make_group_name(prototype_request)
        if self.isfinalized(group=group):
            return

        data = foreal.compute(self.processing_graph, prototype_request)
        ds = data.to_dataset(name=self.dask_key_name)
        ds.to_zarr(self.store, group=group, mode=mode)
        self.isfinalized(True, group=group)
        return

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

        new_request["remove_dependencies"] = True
        new_request["requires_request"] = True

        return new_request

    def isfinalized(self, set_value=None, group=""):
        if not zarr.storage.contains_group(
            self.store
        ) and not zarr.storage.contains_array(self.store):
            return False
        attrs = zarr.open_group(self.store, path=group).attrs

        if set_value is not None:
            attrs["finalized"] = set_value

        if "finalized" in attrs:
            return attrs["finalized"]

        return False

    def forward(self, data, request):
        group = self.make_group_name(request)
        if group not in self.ds:
            with self._mutex:
                if group not in self.ds:
                    self.ds[group] = xr.open_zarr(
                        self.store,
                        group=group,
                        chunk_cache=self.chunk_cache,
                        chunks=None,
                    )
        indexers = request["indexers"]

        section = self.ds[group]
        section = section[self.dask_key_name]
        section = section.sel(indexers_to_slices(indexers))

        return section


import json
import re
import sys
from pathlib import Path
from threading import Lock

import numcodecs
import numpy as np
import zarr
from dask.base import tokenize
from dask.utils import SerializableLock
from numcodecs.blosc import Blosc
from numcodecs.compat import ensure_bytes, ensure_ndarray

import foreal
from foreal.convenience import read_pickle_with_store, to_pickle_with_store
from foreal.core import Node
from foreal.core.graph import NodeFailedException
from typing import Callable
from multiprocessing import Manager

manager = Manager()


def string_timestamp(o):
    if hasattr(o, "isoformat"):
        return o.isoformat()
    else:
        return str(o)


class HashPersister(Node):
    def __init__(
        self,
        store,
        selected_keys=None,
        compression=None,
    ):
        super().__init__()
        if isinstance(store, str) or isinstance(store, Path):
            store = zarr.DirectoryStore(store)
        self.store = store
        self.compression = compression
        self._mutex = SerializableLock()
        # self.what_is_being_written = manager.dict()

        if selected_keys is None:
            # use all keys as hash
            pass

    def isfinalized(finalize=False):
        # TODO: read/write a json `attrs.json` into the same store
        raise NotImplementedError

    def get_hash(self, request):
        # ignore all configs that are meant for hashpersister
        r = {k: v for k, v in request.items() if k != "self"}
        s = json.dumps(r, sort_keys=True, skipkeys=True, default=string_timestamp)
        request_hash = tokenize(s)

        return request_hash

    def is_valid(self, request):
        request_hash = self.get_hash(request)
        # ideally the answer should be yes, no, don't know...
        # currently it's only yes and no

        return request_hash in self.store

    def configure(self, requests=None):
        # merging requests here
        request = super().configure(requests)

        request_hash = self.get_hash(request)

        # forward action defaults to passthrough
        request["self"]["action"] = "passthrough"

        if request["self"].get("bypass", False):
            # set to passthrough -> nothing will happen
            return request

        # propagate the request_hash to the forward function
        request["self"]["request_hash"] = request_hash

        with self._mutex:
            # while holding the mutex, we need to check if the file exists
            if request_hash in self.store:
                # remove previous node since we are going to load from disk
                request["remove_dependencies"] = True

                # set the forward action to load
                request["self"]["action"] = "load"
                return request

            if "fail/" + request_hash in self.store:
                # remove previous node since we are going to load the fail info from disk
                request["remove_dependencies"] = True
                request["self"]["request_hash"] = "fail/" + request_hash

                # set the forward action to load
                request["self"]["action"] = "load"
                return request

            # # check if the file will be written to already
            # if request_hash in self.what_is_being_written:
            #     # yes? that's fine another process handled the same request
            #     # we must tell the system to load the data regularly
            #     # otherwise this node might not get data if the write wasn't
            #     # finished before this node's forward call is processed
            #     # let the system take care of optimizing potential double computations
            #     return request

            # # register that we are going to write to a file
            # self.what_is_being_written[request_hash] = True
            request["self"]["action"] = "store"

        return request

    def forward(self, data, request):
        if request["self"]["action"] == "load":
            data = read_pickle_with_store(
                self.store,
                request["self"]["request_hash"],
                compression=self.compression,
            )
            if isinstance(data,str):
                raise RuntimeError(f'something wrong read {data}')
            return data

        if request["self"]["action"] == "store":
            try:
                # write to file
                # TODO: write to tmp file and move in place
                if isinstance(data, NodeFailedException):
                    to_pickle_with_store(
                        self.store,
                        "fail/" + request["self"]["request_hash"],
                        data,
                        compression=self.compression,
                    )
                else:
                    # data.to_dataset(name=self.dask_key_name).to_zarr(
                    #     self.store,
                    #     group=request["self"]["request_hash"],
                    #     mode="w-",
                    #     compute=False,
                    #     consolidated=True,
                    # )
                    if isinstance(data,str):
                        raise RuntimeError(f'something wrong {data}')
                    to_pickle_with_store(
                        self.store,
                        request["self"]["request_hash"],
                        data,
                        compression=self.compression,
                    )

            except Exception as e:
                print("error", e)
            # finally:
            #     with self._mutex:
            #         # de-register this hash
            #         del self.what_is_being_written[request["self"]["request_hash"]]

            return data
        raise NodeFailedException("A bug in HashPersister. Please report.")


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
                    raise RuntimeError(
                        f"mode `fit` requires that dimension {dim} is in reference {ref}"
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


class ChunkPersister(Node):
    def __init__(
        self,
        store,
        dim: str = "time",
        # classification_scope:dict | Callable[...,dict]=None,
        segment_slice: dict | Callable[..., dict] = None,
        # segment_stride:dict|Callable[...,dict]=None,
        mode: str = "fit",
        ref: dict = None,
    ):
        # if callable(classification_scope):
        #     self.classification_scope = classification_scope
        #     classification_scope = None
        # else:
        #     self.classification_scope = None

        if callable(segment_slice):
            self.segment_slice = segment_slice
            segment_slice = None
        else:
            self.segment_slice = None

        # if callable(segment_stride):
        #     self.segment_stride = segment_stride
        #     segment_stride = None
        # else:
        #     self.segment_stride = None

        super().__init__(
            dim=dim,
            # classification_scope=classification_scope,
            segment_slice=segment_slice,
            # segment_stride=segment_stride,
            mode=mode,
            ref=ref,
        )

        if isinstance(store, str) or isinstance(store, Path):
            store = zarr.DirectoryStore(store)
        self.store = store

    def __dask_tokenize__(self):
        return (ChunkPersister,)

    def configure(self, requests=None):
        request = super().configure(requests)
        rs = request["self"]

        # if rs["dim"] == "index":
        #     indexers = indexers_to_slices(rs["indexers"])["index"]
        #     if isinstance(indexers, slice):
        #         if indexers.start is None or indexers.stop is None:
        #             raise RuntimeError(
        #                 "indexer with dim index must have a start and stop value"
        #             )
        #         indexers = list(
        #             range(indexers.start, indexers.stop, indexers.step or 1)
        #         )
        #     if not isinstance(indexers, list):
        #         raise RuntimeError(
        #             "indexer with dim index must be of type list, dict or slice"
        #         )
        #     segments = [{"index": {"start": x, "stop": x + 1}} for x in indexers]
        # else:

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

        dataset_scope = rs["indexers"]
        segment_slice = get_value("segment_slice")
        # segment_stride = get_value("segment_stride")
        segments = get_segments(
            dataset_scope,
            segment_slice,
            # segment_stride,
            ref=rs["ref"],
            mode=rs["mode"],
            timestamps_as_strings=True,
            minimal_number_of_segments=1,
        )

        cloned_requests = []
        cloned_hashpersisters = []
        for segment in segments:
            segment_request = deepcopy(request)
            del segment_request["self"]
            dict_update(segment_request, {"indexers": segment})
            cloned_requests += [segment_request]
            cloned_hashpersister = HashPersister(
                self.store,
            )
            cloned_hashpersister.dask_key_name = self.dask_key_name + "_hashpersister"
            cloned_hashpersisters += [cloned_hashpersister.forward]

        # Insert predecessor
        # new_request = {}
        request["clone_dependencies"] = cloned_requests
        request["insert_predecessor"] = cloned_hashpersisters

        return request

    def forward(self, data, request):
        if not isinstance(data, list):
            data = [data]
        # print(len(data))
        def unpack_list(inputlist):
            new_list = []
            for item in inputlist:
                if isinstance(item,list):
                    new_list += unpack_list(item)
                else:
                    new_list += [item]
            return new_list

        data = unpack_list(data)
        success = [d for d in data if not isinstance(d, NodeFailedException)]
        
        if not success:
            failed = [str(d) for d in data if isinstance(d, NodeFailedException)]
            raise RuntimeError(f"Failed to data. Reason: {failed}")
        try:
            for i in range(len(success)):
                if not success[i].name or success[i].name is None:
                    success[i].name = "data"
            merged_dataset = xr.merge(success)
        except:
            print(success)
            raise RuntimeError("oh no")

        data = merged_dataset[success[0].name]

        r = request['self']
        indexers = r['indexers']
        slices = indexers_to_slices(indexers)
        section = data.sel(slices)
        section = exclusive_indexing(section, indexers)

        # print(data)
        return section
