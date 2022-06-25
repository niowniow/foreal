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


import copy

import numpy as np
import xarray as xr
import zarr
from tqdm import tqdm

import foreal
from foreal.convenience import (dict_update, indexers_to_slices, is_in_store,
                                make_json_parsable, requests_from_store,
                                requests_to_store)
from foreal.core import Node, Persister, configuration
from foreal.core.graph import NodeFailedException


class Dataset(Node):
    def __init__(self, requests=None, subset="base", record=False, persist_store=None):
        super().__init__(record=record, subset=subset)

        if requests is None or subset is None:
            self.requests = {}
        else:
            self.requests = {subset: requests}
        self.persist_store = persist_store
        if self.persist_store is not None:
            for subset in self.persist_store:
                self.requests[subset] = requests_from_store(self.persist_store, subset)

    def __dask_tokenize__(self):
        return (Dataset,)

    def __len__(self):
        return len(self.requests.get(self.config["subset"], []))

    def length(self, subset=None):
        if subset is None:
            return len(self)
        return len(self.requests[subset])

    def is_persisted(self, subset=None, store=None):
        if subset is None:
            subset = list(self.requests.keys())
        if store is None:
            store = self.persist_store

        is_persisted_list = [is_in_store(store, s) for s in subset]
        if not is_persisted_list:
            return False
        return all(is_persisted_list)

    def persist(self, store=None):
        if store is None:
            store = self.persist_store

        for subset in self.requests:
            self.requests[subset] = np.array(self.requests[subset])
            requests_to_store(self.persist_store, subset, self.requests[subset])

    def configure(self, requests):
        request = super().configure(requests)
        rself = request["self"]

        new_request = {"requires_request": True}
        new_request.update(request)

        if rself.get("bypass", False):
            return new_request

        rc = dict(request)

        # 1. check if this is a recording run
        if request["self"]["record"]:
            # 2. store it internally in self.requests
            recorded_request = copy.deepcopy(request)
            del recorded_request["self"]
            if rself["subset"] not in self.requests:
                self.requests[rself["subset"]] = []
            self.requests[rself["subset"]].append(recorded_request)

            # 3. add remove_dependencies to request
            new_request["remove_dependencies"] = True

            # TODO: 4. store request id in request and delete in forward pass if it fails
            return new_request

        slices = indexers_to_slices(rc["indexers"])
        if "index" in slices:
            # index_request = np.array(self.requests[rself['subset']])[slices['index']].tolist()
            if isinstance(
                slices["index"], list
            ):  # TODO: change `list` to iteratable types
                index_request = [
                    self.requests[rself["subset"]][i] for i in slices["index"]
                ]
            else:
                index_request = self.requests[rself["subset"]][slices["index"]]
            # if len(index_request)>1:
            #     raise RuntimeError('Dataset cannot serve more than one request at a time')

            new_request["clone_dependencies"] = index_request
        return new_request

    def forward(self, data, request):
        if request["self"].get("bypass", False):
            return data

        if request["self"]["record"]:
            # You should not call foreal.compute
            raise NodeFailedException(
                "This is a request recording run. No data can be propagated. You should not call `foreal.compute` but only `foreal.core.configuration`"
            )

        data = data[0]
        # data = xr.concat(data,'index',coords='minimal',compat='override')
        if isinstance(data, xr.DataArray):
            data = data.expand_dims("index")
            index = request["self"]["indexers"]["index"]
            if isinstance(index, dict):
                start = request["self"]["indexers"]["index"]["start"]
                stop = request["self"]["indexers"]["index"]["stop"]
                index = np.arange(start, stop)
            data = data.assign_coords({"index": np.array(index).astype(np.float)})
        elif isinstance(data, np.ndarray):
            data = np.expand_dims(data, axis=0)

        return data


from multiprocessing import Manager

import numpy as np
from tqdm import tqdm

import foreal
from foreal.convenience import dict_update
from foreal.core.graph import NodeFailedException

manager = Manager()


class ForealDatasetHash:
    def __init__(
        self,
        datasets,
        x_datasets,
        persister=None,
        transforms=None,
        subset=None,
        request_base=None,
    ):
        """Wrapper to change a foreal dataset (accessed with requests) into a
        (foreal) dataset accessed with integers.


        Args:
            datasets (Dataset or list): One or many datasets instances to be wrapped. Note: if more than
               one dataset is given all datasets must have the same requests stored
            x_datasets (dask.delayed or list): Node of dataset in the processing_graph. If a list it must have the same length as `datasets`
            transforms (callable or list, optional): Transformation applied to the data after it was loaded. Defaults to None.
            request_base (dict, optional): Request containing basic configurations which will be added to the request for each element of the dataset. If None it defaults to {}.

        Raises:
            RuntimeError: If number of datasets is unqueal to x_datasets
        """
        self.singleton = False
        if not isinstance(datasets, list):
            self.singleton = True
            datasets = [datasets]

        if not isinstance(x_datasets, list):
            x_datasets = [x_datasets]

        if not isinstance(persister, list):
            persister = [persister]

        if len(datasets) != len(x_datasets):
            raise RuntimeError(
                "Make sure the number of datasets instances matches the number of data graphs"
            )

        if request_base is None:
            request_base = {}

        self.datasets = datasets
        self.x_datasets = x_datasets
        self.persister = persister
        self.subset = subset
        self.request_base = request_base

        self.indices = np.arange(datasets[0].length(subset)).tolist()
        # self.indices = {k:k for k in np.arange(datasets[0].length(subset))}

        # a shared dictionary to remember which data items failed to load
        # these can then be removed with mask_invalid()
        self.invalid_indices = manager.dict()

        self.transforms = transforms

    @classmethod
    def join(cls, datasets):
        """
        join from a list of ForealDatasets.
        """

        if not isinstance(datasets, list):
            datasets = [datasets]
        dd = []
        xd = []
        ps = []
        for d in datasets:
            dd += d.datasets
            xd += d.x_datasets
            ps += d.persister

        joined = cls(
            dd,
            xd,
            persister=ps,
            subset=datasets[0].subset,
            request_base=datasets[0].request_base,
        )
        joined.transforms = []

        joined.singleton = 0

        joined.indices = None
        for d in datasets:
            #            joined.datasets += d.datasets
            #            joined.x_datasets += d.x_datasets
            if isinstance(d.transforms, list):
                joined.transforms += d.transforms
            else:
                joined.transforms += [d.transforms]
            joined.singleton += 1
            if joined.indices is None:
                joined.indices = set(d.indices)
            else:
                joined.indices = joined.indices & set(d.indices)
        joined.singleton = joined.singleton <= 1
        joined.indices = list(joined.indices)
        return joined

    @property
    def requests(self):
        # FIXME: change hardcoded
        return self.datasets[0].requests["all"][self.indices]

    def mask_invalid(self):
        # to make it faster we work with a shallow copy of the multiprocessing dict
        local_dict = self.invalid_indices.copy()
        self.indices = [x for x in self.indices if x not in local_dict]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        singleton = self.singleton

        stream_select = np.arange(len(self.datasets))
        if isinstance(idx, tuple):
            idx, stream_select = idx
            if not isinstance(stream_select, (list, np.ndarray)):
                singleton = True
                stream_select = [stream_select]

        internal_idx = self.indices[idx]
        request = dict_update(
            self.request_base,
            {"indexers": {"index": {"start": internal_idx, "stop": internal_idx + 1}}},
        )
        out = []
        for stream in stream_select:
            # TODO: check if dataset was persisted before
            if True:
                values = foreal.compute(self.x_datasets[stream], request)
            else:
                # a faster way to load the data skipping the configuration of the
                # whole graph. It only works if it was persisted before
                r = self.datasets[stream].configure([request])
                if self.persister[stream] is not None:
                    for rc in r["clone_dependencies"]:
                        #                   is_valid = self.persister[stream].is_valid(rc)
                        # print('isvalid',is_valid)
                        rcp = self.persister[stream].configure([rc])
                        if rcp["self"]["action"] == "load":
                            values = self.persister[stream].forward(None, rcp)
                        else:
                            raise RuntimeError("we should not come here")

            if isinstance(values, NodeFailedException):
                self.invalid_indices[int(internal_idx)] = True

            if isinstance(self.transforms, list):
                if self.transforms[stream] is not None:
                    values = self.transforms[stream](values)
            elif self.transforms is not None:
                values = self.transforms(values)

            out.append(values)

        if singleton:
            return out[0]

        return tuple(out)

    def preload(self, batch_size=1, num_workers=0, use_torch=True, client=None):
        if use_torch:
            temp_transforms = self.transforms
            self.transforms = None
            import torch

            for item in tqdm(
                torch.utils.data.dataloader.DataLoader(
                    self,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_last=False,
                    shuffle=False,
                    collate_fn=lambda x: [],
                    # multiprocessing_context=mp.get_context('fork')
                )
            ):
                continue

            self.transforms = temp_transforms

        # # when we get here all dataset_persisters should be intialized
        # for di in range(len(self.dataset_persisters)):
        #     if not use_torch:
        #         group = get_group_from_persister(self.dataset_persisters[di],self.request_base)
        #         if not self.dataset_persisters[di].isfinalized(group=group):
        #             for i in tqdm(range(0, len(self._requests), batch_size)):
        #                 # d = {
        #                 #     "indexers": {"index": {"start": i, "stop": i + batch_size}}
        #                 # }
        #                 d = dict_update(self.request_base,{"indexers": {"index": {"start": i, "stop": i + batch_size}}})

        #                 if client is None:
        #                     foreal.compute(self.x_dataset_persists[di], d)
        #                 else:
        #                     g=foreal.core.configuration(self.x_dataset_persists[di],d)
        #                     out = client.compute(g,sync=True)

        #             # tell the persister that it's finalized. It'll trigger some internal
        #             # optimizations for faster data loading
        #             self.dataset_persisters[di].isfinalized(True,group=group)

        #             # store request in persister

        #             # attrs = zarr.open_group(
        #             #     self.dataset_persisters[di].store, mode="r+", path=group,
        #             # ).attrs
        #             # attrs["requests"] = make_json_parsable(list(self._requests))
        #         else:
        #             if group in self.datasets[di].requests:
        #                 self._requests = self.datasets[di].requests[group]
        #         #     # load request from persister store
        #         #     attrs = zarr.open_group(
        #         #         self.dataset_persisters[di].store, mode="r+", path=group,
        #         #     ).attrs
        #         #     if "requests" in attrs:
        #         #         self._requests = np.array(attrs["requests"])


def get_group_from_persister(persister, request):
    merged_request = persister.merge_config(request)["self"]
    group = persister.make_group_name(merged_request)

    return group


class ForealDataset:
    def __init__(
        self,
        data_graphs,
        dataset_requests,
        filenames,
        transforms=None,
        prototype_index=0,
        persist_dask_key_name=None,
        raw=True,
        bypass_store=False,
        request_base={},
    ):
        """Create dataset by using one or multiple foreal graphs and a set of requests"""
        self.singleton = False
        if not isinstance(data_graphs, list):
            self.singleton = True
            data_graphs = [data_graphs]

        if not isinstance(filenames, list):
            filenames = [filenames]

        if not isinstance(persist_dask_key_name, list):
            persist_dask_key_name = [persist_dask_key_name]

        if not isinstance(raw, list):
            raw = [raw] * len(data_graphs)

        if not isinstance(bypass_store, list):
            bypass_store = [bypass_store] * len(data_graphs)

        if len(data_graphs) != len(filenames):
            raise RuntimeError(
                "Make sure the number of `filenames` matches the number of data graphs"
            )

        if len(data_graphs) != len(persist_dask_key_name):
            raise RuntimeError(
                "Make sure the number of `persist_dask_key_name` matches the number of data graphs"
            )

        # if not data_graphs:
        #     raise RuntimeError('Provide at least one data_graph')

        self.request_base = request_base

        self._requests = dataset_requests
        self.indices = np.arange(len(self._requests))
        prototype_request = dict_update(
            self.request_base,
            {
                "indexers": {
                    "index": {"start": prototype_index, "stop": prototype_index + 1}
                }
            },
        )

        prototype_request = {
            "indexers": {
                "index": {"start": prototype_index, "stop": prototype_index + 1}
            }
        }
        dataset_scope = {
            "index": {"start": 0, "stop": len(dataset_requests)},
        }

        self.datasets = []
        self.dataset_persisters = []
        self.x_datasets = []
        self.x_dataset_persists = []
        # self.data_prototypes = []
        for i, dg in enumerate(data_graphs):
            dataset_persister = Persister(
                filenames[i],
                dataset_scope,
                prototype_request,
                dynamic_dim="index",
                chunk_size=1,
                synchronizer=zarr.ThreadSynchronizer(),
                raw=raw[i],
                bypass_storage=bypass_store[i],
            )

            dataset = Dataset(dataset_requests)

            with foreal.use_delayed():
                x_dataset = dataset(dg)
                x_dataset_persist = dataset_persister(
                    x_dataset, dask_key_name=persist_dask_key_name[i]
                )

            if not bypass_store[i]:
                data_prototype = dataset_persister.initialize(
                    processing_graph=x_dataset, step=1
                )

            # self.data_prototypes += [data_prototype]
            self.datasets += [dataset]
            self.dataset_persisters += [dataset_persister]
            self.x_datasets += [x_dataset]
            self.x_dataset_persists += [x_dataset_persist]

        self.transforms = transforms

    @classmethod
    def from_components(
        cls,
        dataset,
        dataset_persister,
        x_dataset,
        x_dataset_persist,
        requests=None,
        transform=None,
    ):
        fd = cls([], [], [], persist_dask_key_name=[])
        fd.datasets = [dataset]
        fd.dataset_persisters = [dataset_persister]
        fd.x_datasets = [x_dataset]
        fd.x_dataset_persists = [x_dataset_persist]
        fd.transforms = [transform]
        fd._requests = requests
        if requests is None:
            fd._requests = dataset.requests[dataset.config["subset"]]
        elif isinstance(requests, str):
            fd._requests = dataset.requests[requests]

        fd.indices = np.arange(len(fd._requests))
        return fd

    @classmethod
    def join(cls, datasets):
        """
        join from a list of ForealDatasets.
        """

        if not isinstance(datasets, list):
            datasets = [datasets]

        joined = cls([], [], [], persist_dask_key_name=[])
        joined.transforms = []
        joined._requests = copy.copy(datasets[0]._requests)
        joined.indices = np.arange(
            len(joined._requests)
        )  # TODO: is intended behaviour?
        # joined.indices = copy.copy(datasets[0].indices) # TODO: or this?

        joined.singleton = 0

        for d in datasets:
            joined.datasets += d.datasets
            joined.dataset_persisters += d.dataset_persisters
            joined.x_datasets += d.x_datasets
            joined.x_dataset_persists += d.x_dataset_persists
            if isinstance(d.transforms, list):
                joined.transforms += d.transforms
            else:
                joined.transforms += [d.transforms]
            joined.singleton += 1

        joined.singleton = joined.singleton <= 1

        return joined

    @property
    def requests(self):
        return self._requests[self.indices]

    def valid(self, idx, stream_select=None):
        stream_indices = np.arange(len(self.dataset_persisters))
        if stream_select is not None:
            if not isinstance(stream_select, (list, np.ndarray)):
                stream_select = [stream_select]
            stream_indices = stream_select
        is_valid = True
        for i in stream_indices:
            # request = {"indexers": {"index": {"start": idx, "stop": idx + 1}}}
            request = dict_update(
                self.request_base,
                {"indexers": {"index": {"start": idx, "stop": idx + 1}}},
            )

            is_valid = (
                is_valid
                and (
                    self.dataset_persisters[i].isavailable(request, merge_config=True)
                    == 1
                ).all()
            )

        return is_valid

    def mask_invalid(self):
        # request = {"indexers": {"index": {"start": 0, "stop": len(self)}}}
        request = dict_update(
            self.request_base, {"indexers": {"index": {"start": 0, "stop": len(self)}}}
        )

        all_valid = None
        for i in range(len(self.dataset_persisters)):
            valid = (
                self.dataset_persisters[i].isavailable(request, merge_config=True) == 1
            )

            if all_valid is None:
                all_valid = valid
            else:
                all_valid = np.logical_and(all_valid, valid)

        self.indices = np.nonzero(all_valid)[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        singleton = self.singleton

        stream_select = np.arange(len(self.dataset_persisters))
        if isinstance(idx, tuple):
            idx, stream_select = idx
            if not isinstance(stream_select, (list, np.ndarray)):
                singleton = True
                stream_select = [stream_select]

        idx = self.indices[idx]
        # request = {"indexers": {"index": {"start": idx, "stop": idx + 1}}}
        request = dict_update(
            self.request_base, {"indexers": {"index": {"start": idx, "stop": idx + 1}}}
        )
        out = []
        for i in stream_select:
            group = get_group_from_persister(
                self.dataset_persisters[i], self.request_base
            )
            if self.dataset_persisters[i].isfinalized(
                group=group
            ) and not self.dataset_persisters[i].config.get("bypass_storage", False):
                # it's much faster to retrieve the values without configuration
                values = self.dataset_persisters[i](request=request)
            else:
                # we need to configure&compute because the item might not be loaded
                if self.dataset_persisters[i].config.get("bypass_storage", False):
                    values = foreal.compute(self.x_datasets[i], request)
                else:
                    values = foreal.compute(self.x_dataset_persists[i], request)

            if isinstance(self.transforms, list):
                if self.transforms[i] is not None:
                    values = self.transforms[i](values)
            elif self.transforms is not None:
                values = self.transforms(values)

            out.append(values)

        if singleton:
            return out[0]

        return tuple(out)

    def preload(self, batch_size=1, num_workers=0, use_torch=False, client=None):
        # preloading by loading the whole dataset once
        # since we introduced a persister, the data will be stored to disk
        # and when we load it again later it will be fetched from disk
        #        if self.bypass_store:
        #            raise RuntimeError("cannot init store if its bypassed")

        if use_torch:
            temp_transforms = self.transforms
            self.transforms = None
            import torch

            for di in range(len(self.dataset_persisters)):
                group = get_group_from_persister(
                    self.dataset_persisters[di], self.request_base
                )
                if not self.dataset_persisters[di].isfinalized(group=group):
                    for item in tqdm(
                        torch.utils.data.dataloader.DataLoader(
                            self,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=False,
                            shuffle=False,
                            collate_fn=lambda x: [],
                            # multiprocessing_context=mp.get_context('fork')
                        )
                    ):
                        continue

                    # cal the above only once, since dataloader call __getitem__ which loads
                    # all dataset_persisters
                    break
            self.transforms = temp_transforms

        # when we get here all dataset_persisters should be intialized
        for di in range(len(self.dataset_persisters)):
            if not use_torch:
                group = get_group_from_persister(
                    self.dataset_persisters[di], self.request_base
                )
                if not self.dataset_persisters[di].isfinalized(group=group):
                    for i in tqdm(range(0, len(self._requests), batch_size)):
                        # d = {
                        #     "indexers": {"index": {"start": i, "stop": i + batch_size}}
                        # }
                        d = dict_update(
                            self.request_base,
                            {
                                "indexers": {
                                    "index": {"start": i, "stop": i + batch_size}
                                }
                            },
                        )

                        if client is None:
                            foreal.compute(self.x_dataset_persists[di], d)
                        else:
                            g = foreal.core.configuration(
                                self.x_dataset_persists[di], d
                            )
                            out = client.compute(g, sync=True)

                    # tell the persister that it's finalized. It'll trigger some internal
                    # optimizations for faster data loading
                    self.dataset_persisters[di].isfinalized(True, group=group)

                    # store request in persister

                    # attrs = zarr.open_group(
                    #     self.dataset_persisters[di].store, mode="r+", path=group,
                    # ).attrs
                    # attrs["requests"] = make_json_parsable(list(self._requests))
                else:
                    if group in self.datasets[di].requests:
                        self._requests = self.datasets[di].requests[group]
                #     # load request from persister store
                #     attrs = zarr.open_group(
                #         self.dataset_persisters[di].store, mode="r+", path=group,
                #     ).attrs
                #     if "requests" in attrs:
                #         self._requests = np.array(attrs["requests"])
