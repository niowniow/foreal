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

import datetime as dt
import os
import warnings
from os.path import abspath, dirname, join

import appdirs
import yaml

# initialize global settings with certain default values
_GLOBAL_CONFIG_DICT = {
    "reference_time": dt.datetime(2000, 1, 1, 0, 0, 0),
    "datetime_format": None,
    "fail_mode": "fail",
    "use_delayed": False,
    "probe_request": None,
}


def get_global_config():
    return _GLOBAL_CONFIG_DICT


def setting_exists(key):
    return key in _GLOBAL_CONFIG_DICT


def get_setting(key):
    """Returns the global setting for the given key

    Arguments:
        key {string} -- setting name

    Returns:
        [type] -- setting value
    """

    try:
        return _GLOBAL_CONFIG_DICT[key]
    except KeyError:
        print(
            f"""The foreal setting {key} was not found and is required by a function call.
        Set it before your first call to the foreal package.
        This can be done by either providing a settings file via foreal.load_config(),
        updating your user config file or updating the settings directly with
               foreal.set_setting('{key}',value)"""
        )


def get_setting_path(key):
    """Get global setting and make sure it is a valid path

    Arguments:
        key {string} -- setting name

    Returns:
        [type] -- setting value
    """
    path = join(get_setting(key), "")
    if not os.path.isdir(path):
        warnings.warn("foreal requested a path which is invalid: {}".format(path))
    return path


def set_setting(key, value):
    _GLOBAL_CONFIG_DICT[key] = value


def load_config(filename):
    """Load settings from a yaml file.

    This is used to setup the workspace, add required keys and make the module
    be able to find the necessary directories

    The settings file can for example contain the following items:

    ```python
        data_dir: '/path/to/data_dir/'
        user_dir: '/path/to/user_dir/'
    ```

    """

    if os.path.isfile(filename):
        with open(filename, "r") as f:
            settings = yaml.safe_load(f)
    else:
        raise IOError("Parameter file not found [%s]" % filename)

    _GLOBAL_CONFIG_DICT.update(settings)


def get_user_config_file():
    return join(appdirs.AppDirs("foreal", "foreal").user_config_dir, "config.yml")


def create_user_config_file():
    filename = get_user_config_file()
    if os.path.isfile(filename):
        print("User config file already exists at", filename)
        return

    os.makedirs(dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write("")
    print("User config file created at", filename)


def load_user_config():
    """Check if a user specific config exists and load it"""

    user_config = get_user_config_file()

    if os.path.isfile(user_config):
        load_config(user_config)


# class set:
#     def __init__(self,setting,value):
#         self.prev = None
#         self.setting = setting
#         self.value = value
#         pass

#     def __enter__(self):
#         if setting_exists(self.setting):
#             self.prev = get_setting(self.setting)
#         set_setting(self.setting, self.value)

#     def __exit__(self, type, value, traceback):
#         if self.prev is None:
#             set_setting(self.setting, self.prev)

# FIXME: it could happen that this is called during
# runtime and overwrites all custom configurations!
load_user_config()  # initially load the user settings
