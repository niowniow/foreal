from copy import deepcopy
from pathlib import Path

import dask
import numpy as np
import pandas as pd

import foreal
from foreal.apps.seismic import SeismicPortal
from foreal.config import get_setting, set_setting
from foreal.convenience import indexers_to_slices
from foreal.core import Node, Persister, requests
from foreal.core.graph import NestedFrozenDict
from foreal.processing import MinMaxDownsampling

# test_data_dir = Path(__file__).absolute().parent.joinpath("data")
test_data_dir = Path(get_setting("persist_dir")) / "tests/test_persist/"
print("Test data stored at", test_data_dir)


def test_persister():
    permasense_vault = get_setting(
        "permasense_vault_dir"
    )  # TODO: create separate repository to store the test data
    data_path = Path(permasense_vault) / "datasets/illgraben/XP/"
    seismic_store = foreal.DirectoryStore(data_path)

    # TODO: write a fake data source to remove dependency on actual datasets!
    seismic_source_continuous_node = SeismicPortal(
        store=seismic_store,
        channel=["EHE", "EHN", "EHZ"],
        # use_arclink=True,  # station='MH36',
        network="XP",
        location="",
        remove_response=False,
    )
    dataset_scope = {
        "time": {"start": "2017-05-01", "stop": "2018-01-01"},
        "channel": ["XP.ILL03..EHE", "XP.ILL03..EHN", "XP.ILL03..EHZ"],
    }
    prototype_request = {
        "indexers": {
            "time": {
                "start": "2017-06-28T06:00:00.000",
                "stop": "2017-06-28T07:00:00.000",
            }
        },
        "station": "ILL08",
        "network": "XP",
        "location": "",
    }
    persister = Persister(
        test_data_dir / "waveform.zarr",
        dataset_scope,
        prototype_request,
        group_keys=["station"],
    )

    with foreal.use_delayed():
        x_source = seismic_source_continuous_node(dask_key_name="source")

        x_p_in = x_source
        sampling_rate = 100
        step = pd.to_timedelta(1 / sampling_rate, "s")
        x_persist = persister(x_p_in, dask_key_name="persist")
        persister.initialize(processing_graph=x_p_in, step=step, mode="w")

    # Compare the prototype request
    def compare_request(request, raw=False):
        data = foreal.core.configuration(x_p_in, request)
        with dask.config.set(scheduler="single-threaded"):
            base_data = dask.compute(data)[0]

        data = foreal.core.configuration(x_persist, request)
        with dask.config.set(scheduler="single-threaded"):
            persist_data = dask.compute(data)[0]

        if not raw:
            persist_data = persist_data.compute()

        def correct_frame_extracted(data, request):
            start_cond = data["time"] >= foreal.to_datetime(
                request["indexers"]["time"]["start"]
            )
            stop_cond = data["time"] < foreal.to_datetime(
                request["indexers"]["time"]["stop"]
            )
            return bool(start_cond.all() and stop_cond.all())

        # base_cond = correct_frame_extracted(base_data, request)
        # persist_cond = correct_frame_extracted(persist_data, request)
        # print(base_data)
        # print(persist_data)
        if not raw:
            assert np.array_equal(base_data.values, persist_data.values)
            assert base_data.equals(persist_data)
        else:
            assert isinstance(persist_data, np.ndarray)
            assert np.array_equal(base_data.values, persist_data)

    compare_request(prototype_request)

    # compare outside
    prototype_request = {
        "indexers": {
            "time": {
                "start": "2017-06-28T08:04:23.124",
                "stop": "2017-06-28T09:01:00.000",
            }
        },
        "station": "ILL08",
        "network": "XP",
        "location": "",
    }
    compare_request(prototype_request)

    # compare overlap
    prototype_request = {
        "indexers": {
            "time": {
                "start": "2017-06-28T05:04:23.124",
                "stop": "2017-06-28T09:12:00.000",
            }
        },
        "station": "ILL08",
        "network": "XP",
        "location": "",
    }
    compare_request(prototype_request)

    # compare bypass
    prototype_request = {
        "indexers": {
            "time": {
                "start": "2017-06-28T05:04:23.124",
                "stop": "2017-06-28T09:12:00.000",
            }
        },
        "station": "ILL08",
        "network": "XP",
        "location": "",
        "bypass": True,
    }
    compare_request(prototype_request)

    # config (global)
    prototype_request = {
        "indexers": {
            "time": {
                "start": "2017-06-28T05:04:23.124",
                "stop": "2017-06-28T09:12:00.000",
            }
        },
        "config": {"global": {"station": "ILL08", "network": "XP", "location": ""}},
    }
    compare_request(prototype_request)

    # config (keys)
    prototype_request = {
        "indexers": {
            "time": {
                "start": "2017-06-28T05:04:23.124",
                "stop": "2017-06-28T09:12:00.000",
            }
        },
        "config": {
            "global": {"station": "ILL08", "network": "XP", "location": ""},
            "keys": {"persist": {"raw": True}},
        },
    }
    compare_request(prototype_request, raw=True)

    # config (type)
    prototype_request = {
        "indexers": {
            "time": {
                "start": "2017-06-28T05:04:23.124",
                "stop": "2017-06-28T09:12:00.000",
            }
        },
        "config": {
            "global": {"station": "ILL08", "network": "XP", "location": ""},
            "types": {"Persister": {"raw": True}},
        },
    }
    compare_request(prototype_request, raw=True)

    prototype_request = {
        "indexers": {
            "time": {
                "start": "2017-06-28T06:00:00.000",
                "stop": "2017-06-28T07:00:00.000",
            }
        },
        "station": "ILL08",
        "network": "XP",
        "location": "",
    }

    # create persister with default raw
    persister = Persister(
        test_data_dir / "waveform.zarr",
        dataset_scope,
        prototype_request,
        group_keys=["station"],
        raw=True,
    )

    with foreal.use_delayed():
        x_source = seismic_source_continuous_node(dask_key_name="source")

        x_persist = persister(x_source, dask_key_name="persist")
        persister.initialize(processing_graph=x_source, step=step, mode="w")

    # compare outside
    prototype_request = {
        "indexers": {
            "time": {
                "start": "2017-06-28T08:04:23.124",
                "stop": "2017-06-28T09:01:00.000",
            }
        },
        "station": "ILL08",
        "network": "XP",
        "location": "",
    }
    compare_request(prototype_request, raw=True)


# FIXME: remove dependency of Dataset()
# def test_persister_index():
#     permasense_vault = get_setting(
#         "permasense_vault_dir"
#     )  # TODO: create separate repository to store the test data
#     data_path = Path(permasense_vault) / "datasets/illgraben/XP/"
#     seismic_store = foreal.DirectoryStore(data_path)

#     prototype_request = {
#         "indexers": {
#             'index':{'start':0,'stop':1}, #FIXME: remove stop
#             "time": {
#                 "start": "2017-06-28T08:04:23.124",
#                 "stop": "2017-06-28T09:01:00.000",
#             }
#         },
#         "station": "ILL08",
#         "network": "XP",
#         "location": "",
#     }


#     # TODO: write a fake data source to remove dependency on actual datasets!
#     seismic_source_continuous_node = SeismicPortal(
#         store=seismic_store,
#         channel=["EHE", "EHN", "EHZ"],
#         # use_arclink=True,  # station='MH36',
#         network="XP",
#         location="",
#         remove_response=False,
#     )
#     dataset_scope = {
#         "index": {'start':0,'stop':10},
#         "time": {"start": "2017-05-01", "stop": "2018-01-01"},
#         "channel": ["XP.ILL03..EHE", "XP.ILL03..EHN", "XP.ILL03..EHZ"],
#     }
#     prototype_request = {
#         "indexers": {
#             "time": {
#                 "start": "2017-06-28T06:00:00.000",
#                 "stop": "2017-06-28T07:00:00.000",
#             }
#         },
#         "station": "ILL08",
#         "network": "XP",
#         "location": "",
#     }


#     requests = []
#     for i in range(dataset_scope['index']['start'],dataset_scope['index']['stop']):
#         request = deepcopy(prototype_request)
#         # request['indexers']['index'] = {'start':i,'stop':i+1}
#         request['indexers']['time']['start'] = (pd.to_datetime(request['indexers']['time']['start']) + i*pd.to_timedelta('30s')).isoformat()
#         request['indexers']['time']['stop'] = (pd.to_datetime(request['indexers']['time']['stop']) + i*pd.to_timedelta('30s')).isoformat()
#         requests.append(request)


#     import xarray as xr


#     prototype_request = {
#         "indexers": {
#             'index':{'start':0,'stop':1},
#         },
#     }

#     persister = Persister(
#         test_data_dir / "persist_index.zarr",
#         dataset_scope,
#         prototype_request,
#         dynamic_dim = 'index',
#         chunk_size=1,
#     )

#     def time_to_delta(x):
#         x['time'] = x['time'] - x['time'][0]
#         return x
#     with foreal.use_delayed():
#         x_source = seismic_source_continuous_node(dask_key_name="source")
#         x_dataset = Dataset(requests)(x_source)
#         x_delta = foreal.it(time_to_delta)(x_dataset)
#         x_persist = persister(x_delta, dask_key_name="persist")
#         persister.initialize(
#             processing_graph=x_delta, step=1, mode="w"
#         )

#     # Compare the prototype request
#     def compare_request(request):
#         base_data = foreal.compute(x_delta,request)
#         persist_data = foreal.compute(x_persist,request)

#         def correct_frame_extracted(data, request):
#             start_cond = data["time"] >= foreal.to_datetime(
#                 request["indexers"]["time"]["start"]
#             )
#             stop_cond = data["time"] < foreal.to_datetime(
#                 request["indexers"]["time"]["stop"]
#             )
#             return bool(start_cond.all() and stop_cond.all())

#         # base_cond = correct_frame_extracted(base_data, request)
#         # persist_cond = correct_frame_extracted(persist_data, request)
#         # print(f'{base_data=}')
#         # print(f'{base_data.values=}')
#         # print(f'{persist_data=}')
#         # print(f'{persist_data.values=}')
#         assert np.array_equal(base_data.values, persist_data.values)
#         assert base_data.equals(persist_data)

#     # compare outside
#     prototype_request = {
#         "indexers": {
#             'index':{'start':0,'stop':1},
#         },
#     }
#     compare_request(prototype_request)


#     prototype_request = {
#         "indexers": {
#             'index':{'start':5,'stop':6},
#         },
#     }
#     compare_request(prototype_request)


#     # prototype_request = {
#     #     "indexers": {
#     #         'index':{'start':11},
#     #     },
#     #     'station': 'ILL08',
#     # }
#     # compare_request(prototype_request)


def test_faulty_data():
    permasense_vault = get_setting(
        "permasense_vault_dir"
    )  # TODO: create separate repository to store the test data
    data_path = Path(permasense_vault) / "datasets/illgraben/XP/"
    seismic_store = foreal.DirectoryStore(data_path)

    # TODO: write a fake data source to remove dependency on actual datasets!
    seismic_source_continuous_node = SeismicPortal(
        store=seismic_store,
        channel=["EHE", "EHN", "EHZ"],
        # use_arclink=True,  # station='MH36',
        network="XP",
        location="",
        remove_response=False,
    )
    dataset_scope = {
        "time": {"start": "2017-05-01", "stop": "2018-01-01"},
        "channel": ["XP.ILL03..EHE", "XP.ILL03..EHN", "XP.ILL03..EHZ"],
    }
    prototype_request = {
        "indexers": {
            "time": {
                "start": "2017-06-28T06:00:00.000",
                "stop": "2017-06-28T07:00:00.000",
            }
        },
        "station": "ILL08",
        "network": "XP",
        "location": "",
    }
    persister = Persister(
        test_data_dir / "waveform.zarr",
        dataset_scope,
        prototype_request,
        group_keys=["station"],
    )

    def faultify(da):
        if not hasattr(faultify, "mean"):
            faultify.mean = da.mean("time")
        offset = (da - faultify.mean).sum("channel")
        rn = np.broadcast_to(offset, da.shape)
        faulty = da.where(offset > 0, drop=True)
        return faulty

    with foreal.use_delayed():
        x_source = seismic_source_continuous_node(dask_key_name="source")

        x_fault = foreal.it(faultify)(x_source)

        x_p_in = x_source
        sampling_rate = 100
        # sampling_rate=100/2/2/2
        x_persist = persister(x_fault, dask_key_name="persist")
        persister.initialize(
            processing_graph=x_p_in, sampling_rate=sampling_rate, mode="w"
        )

    # Compare the prototype request
    def compare_request(request):
        data = foreal.core.configuration(x_p_in, request)
        with dask.config.set(scheduler="single-threaded"):
            base_data = dask.compute(data)[0]

        data = foreal.core.configuration(x_fault, request)
        with dask.config.set(scheduler="single-threaded"):
            faulty_data = dask.compute(data)[0]

        data = foreal.core.configuration(x_persist, request)
        with dask.config.set(scheduler="single-threaded"):
            persist_data = dask.compute(data)[0]
        persist_data = persist_data.compute()

        persist_data = faultify(persist_data)

        assert np.array_equal(faulty_data.values, persist_data.values)
        assert faulty_data.equals(persist_data)

    compare_request(prototype_request)

    # compare outside
    prototype_request = {
        "indexers": {
            "time": {
                "start": "2017-06-28T08:04:23.124",
                "stop": "2017-06-28T09:01:00.000",
            }
        },
        "station": "ILL08",
        "network": "XP",
        "location": "",
    }
    compare_request(prototype_request)

    # compare overlap
    prototype_request = {
        "indexers": {
            "time": {
                "start": "2017-06-28T05:04:23.124",
                "stop": "2017-06-28T09:12:00.000",
            }
        },
        "station": "ILL08",
        "network": "XP",
        "location": "",
    }
    compare_request(prototype_request)
