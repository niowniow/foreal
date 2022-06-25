from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr

import foreal
from foreal.apps.seismic import SeismicPortal
from foreal.config import get_setting
from foreal.processing import MinMaxDownsampling, Spectrogram


def test_minmax():
    input_sampling_rate = pd.to_timedelta(1, "s") / pd.to_timedelta(1, "D")
    minmax_default = MinMaxDownsampling(dim="time")
    minmax_rate2 = MinMaxDownsampling(
        rate=2, input_sampling_rate=input_sampling_rate, dim="time"
    )

    np.random.seed(123)

    length = 9
    data = np.arange(length * 2).reshape((length, 2))
    # data = np.random.rand(length, 2)
    da = xr.DataArray(
        data,
        [
            ("time", pd.date_range("2000-01-01", periods=length)),
            ("channel", ["EHE", "EHZ"]),
        ],
    )

    from numpy.lib.stride_tricks import sliding_window_view

    print(data)
    x = sliding_window_view(data, (4, 2))[::4]
    print(x)
    print(x.shape)
    x_min = x.min(axis=-2).squeeze()
    x_max = x.max(axis=-2).squeeze()
    print(x_min)

    expected_data = np.concatenate((x_min, x_max), axis=0)
    expected_data[::2] = x_min
    expected_data[1::2] = x_max
    print(expected_data)
    expected = xr.DataArray(
        expected_data,
        [
            (
                "time",
                pd.date_range(
                    "2000-01-01", freq=pd.to_timedelta(2, "D"), periods=length // 2
                ),
            ),
            ("channel", ["EHE", "EHZ"]),
        ],
    )

    x = minmax_rate2(da)

    print("original", da)
    print("computed", x)
    print("expected", expected)

    print(da["time"].values)
    print(x["time"].values)
    print(expected["time"].values)

    # assert x.shape == (4, 2)
    # assert x.mean() == 0.46291252327532006
    # assert x.equals(expected)


# def test_spectogram():
#     request = {'indexers': {'time': {'start': '2017-01-01T04:13:30', 'stop': '2017-01-01T04:14:00'}, 'channel': ['4D.MH36.A.EHZ']}, 'station': 'MH36'}
#     request_raw = deepcopy(request)
#     request_raw['raw'] = True

#     permasense_vault = get_setting(
#         "permasense_vault_dir"
#     )  # TODO: create separate repository to store the test data
#     data_path = Path(permasense_vault) / "datasets/illgraben/XP/"
#     seismic_store = foreal.DirectoryStore(data_path)
#     sampling_rate = 250
#     # TODO: write a fake data source to remove dependency on actual datasets!
#     seismic_source_continuous_node = SeismicPortal(
#         store=seismic_store,
#         channel=["EHE", "EHN", "EHZ"],
#         # use_arclink=True,  # station='MH36',
#         network="XP",
#         location="",
#         remove_response=False,
#     )
#     spectrogram = Spectrogram(
#         nfft=256, stride=8, dim="time", sampling_rate=sampling_rate, detrend=None,
#     )

#     with foreal.use_delayed():
#         x_source = seismic_source_continuous_node(dask_key_name="source")

#         x_p_in = x_source
#         sampling_rate = 100
#         step = pd.to_timedelta(1 / sampling_rate, "s")
#         x_persist = persister(x_p_in, dask_key_name="persist")
#         persister.initialize(
#             processing_graph=x_p_in, step=step, mode="w"
#         )

#     foreal.compute(x_signal,request)

#     assert data == data.raw

#     assert np.array_equal(base_data.values, persist_data)
