import numpy as np

import foreal
from foreal.apps.models import ModelWrapperGenerator
from foreal.apps.seismic import SeismicPortal
from foreal.processing import Spectrogram


def to_db(x, min_value=1e-20, reference=1.0):
    value_db = 10.0 * np.log10(np.maximum(min_value, x))
    value_db -= 10.0 * np.log10(np.maximum(min_value, reference))
    return value_db


def get_taskgraphs():
    # Create an instance of a seismic portal to load the data from arclink.ethz.ch
    seismic_inst = SeismicPortal(
        use_arclink={"url": "http://eida.ethz.ch"},
        channel=["EHZ"],
        station=["MH36"],
        location=["A"],
        network=["1I"],
    )

    # we want to process the data as a spectrogram. thus we create a spectrogam instance
    # with some default parameters
    spectrogram_inst = Spectrogram(
        nfft=256,
        stride=8,
        dim="time",
    )

    # we will return a dict with all graph endpoints we would like to make available
    # as distinct taskgraphs
    d = {}

    """
foreal has two modes: either the foreal nodes are processed immediatly or delayed.
immediate processing will generate the results when we call the node e.g.
`result = seismic_inst(request={"indexers": {"time":{"start":"2022-06-05T11:20:00","stop":"2022-06-05T11:40:00"}}})`
In contrast, delayed processing will create a taskgraph which will not be computed yet
It just creates the structure of the desired processing. Please also
refer to dask.delayed documentation: https://docs.dask.org/en/stable/delayed.html"""

    # By default each Node is immediate mode. To chain together our taskgraph we can
    # use foreal.use_delayed(). all nodes within the block will be set to delayed mode
    with foreal.use_delayed():
        # now the call of seismic_inst creates a delayed object `seismic_node`.
        seismic_node = seismic_inst(dask_key_name="seismic")
        # `seismic_node` can be used as an input dummy for other computations
        spectrogram_node = spectrogram_inst(seismic_node, dask_key_name="spectrogram")
        # we can make any function into a basic foreal node by using foreal.it()
        spectrogram_db_node = foreal.it(to_db)(spectrogram_node, dask_key_name="to_db")

    d["default"] = spectrogram_db_node

    return d
