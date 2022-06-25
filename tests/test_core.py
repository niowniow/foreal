from pathlib import Path

import dask

import foreal
from foreal.apps.seismic import SeismicPortal
from foreal.config import get_setting, set_setting
from foreal.convenience import dict_update
from foreal.core import Node, Persister
from foreal.core.requests import get_dataset_scopes
from foreal.processing import MinMaxDownsampling

# test_data_dir = Path(__file__).absolute().parent.joinpath("data")
test_data_dir = Path(get_setting("persist_dir")) / "tests/test_persist/"
print(test_data_dir)


from copy import deepcopy


class Chunker(Node):
    def __init__(self, segment_shape, segment_stride, global_reference=None):
        super().__init__(
            segment_shape=segment_shape,
            segment_stride=segment_stride,
            global_reference=global_reference,
        )

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

        if request is not None:
            segment_slice = {"time": foreal.to_timedelta(30, "seconds")}
            segment_stride = {"time": foreal.to_timedelta(30, "seconds")}

            print("assembling request slices")
            slices = get_dataset_scopes(
                segment_slice, request["indexers"], segment_stride
            )
            request_list = [
                dict_update(deepcopy(new_request), {"indexers": indexer})
                for indexer in slices
            ]
            new_request["clone_dependencies"] = request_list
            new_request["requires_request"] = True

        return new_request

    def forward(self, data, request):
        return data


def test_optimize():
    permasense_vault = get_setting(
        "permasense_vault_dir"
    )  # TODO: create separate repository to store the test data
    data_path = Path(permasense_vault) / "datasets/illgraben/XP/"
    sampling_rate = 100

    seismic_store = foreal.DirectoryStore(data_path)

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
        "channel": ["XP.ILL08..EHE", "XP.ILL08..EHN", "XP.ILL08..EHZ"],
        "stream": [0],
    }
    dataset_scope = {
        "time": {"start": "2017-05-01", "stop": "2018-01-01"},
        "channel": ["XP.ILL08..EHE", "XP.ILL08..EHN", "XP.ILL08..EHZ"],
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
        test_data_dir / "test_optimize.zarr",
        dataset_scope,
        prototype_request,
        group_keys=["station"],
    )

    segment_slice = {"time": foreal.to_timedelta(30, "seconds")}
    segment_stride = {"time": foreal.to_timedelta(30, "seconds")}
    test_request = {
        "indexers": {
            "time": {
                "start": "2017-05-24T06:00:00.000",
                "stop": "2017-05-24T06:05:00.000",
            }
        },
        "station": "ILL03",
        "network": "XP",
        "location": "",
    }

    with foreal.use_delayed():
        x_source = seismic_source_continuous_node(dask_key_name="source")
        x_p_in = x_source
        x_persist = persister(x_p_in, dask_key_name="persist")
        persister.initialize(
            processing_graph=x_p_in, sampling_rate=sampling_rate, mode="w"
        )
        x_chunks = Chunker(segment_slice, segment_stride)(
            x_persist, dask_key_name="chunker"
        )

    graph = foreal.core.configuration(x_chunks, test_request, optimize_graph=False)
    # dask.visualize(graph,filename='before.png')

    graph = foreal.core.optimize(graph, dask_optimize=False)
    # dask.visualize(graph,filename='after.png')

    # TODO: add assert when rewrite with FAKE DATASET
