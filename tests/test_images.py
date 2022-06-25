import datetime as dt
from pathlib import Path

import foreal
from foreal.apps.permasense import MHDSLRFilenames, MHDSLRImages
from foreal.config import get_setting, set_setting

channels = ["EHE", "EHN", "EHZ"]
stations = ["MH36", "MH44", "MH48", "MH52", "MH54"]

start_time = dt.datetime(2017, 7, 14, 7, 7, 0, tzinfo=None)
end_time = dt.datetime(2017, 7, 14, 7, 7, 10, tzinfo=None)

offset = dt.timedelta(days=1)
config = {"indexers": {"time": {"start": start_time, "stop": end_time}}}


def test_permasense_image_filenames():

    permasense_vault = get_setting("permasense_vault_dir")
    data_path = Path(permasense_vault) / "datasets/MHDSLR/"

    # first test without config
    node = MHDSLRFilenames(base_directory=data_path)

    start_time = dt.datetime(2017, 8, 6, 9, 56, 12, tzinfo=None)
    end_time = dt.datetime(2017, 8, 6, 10, 14, 10, tzinfo=None)

    offset = dt.timedelta(days=1)
    config_0 = {
        "indexers": {"time": {"start": start_time, "stop": end_time}},
    }

    data = node(request=config_0)

    config_1 = config.copy()
    # this should return and empty list
    data = node(request=config_1)

    # Test if we do not provide a end_time
    del config_0["indexers"]["time"]["stop"]
    data = node(request=config_0)

    del config_1["indexers"]["time"]["stop"]
    data = node(request=config_1)

    config_1["indexers"]["time"]["start"] = dt.datetime(
        2018, 8, 6, 20, 0, 0, tzinfo=None
    )
    data = node(request=config_1)


def test_permasense_mhdslrimage():
    permasense_vault = get_setting("permasense_vault_dir")
    data_path = Path(permasense_vault) / "datasets/MHDSLR/"
    node = MHDSLRImages(base_directory=data_path)

    start_time = dt.datetime(2017, 8, 6, 9, 50, 12, tzinfo=None)
    end_time = dt.datetime(2017, 8, 6, 10, 12, 10, tzinfo=None)

    offset = dt.timedelta(days=1)
    config = {
        "indexers": {"time": {"start": start_time, "stop": end_time}},
    }

    data = node(request=config)

    config["output_format"] = "base64"
    data = node(request=config)

    import base64

    from PIL import Image

    with open(
        data_path / "JPG" / "2017-08-06" / "20170806_095212.JPG", "rb"
    ) as image_file:
        img_base64 = base64.b64encode(image_file.read())
        assert data[0].values == img_base64

    # Check a period where there is no image
    start_time = dt.datetime(2010, 8, 6, 9, 55, 12, tzinfo=None)
    end_time = dt.datetime(2010, 8, 6, 10, 10, 10, tzinfo=None)
    config = {
        "indexers": {"time": {"start": start_time, "stop": end_time}},
    }
    data = node(request=config)

    # print(data)
    assert data.shape == (0, 0, 0, 0)
