import numpy as np
import pandas as pd
import xarray as xr

import foreal
from foreal.apps.annotations import BoundingBoxAnnotation, requests_from_file
from foreal.apps.models import ModelWrapperGenerator
from foreal.apps.seismic import SeismicPortal
from foreal.config import get_setting
from foreal.core.datasets import Dataset
from foreal.core.persist import HashPersister
from foreal.processing import Spectrogram


def to_db(x, min_value=1e-20, reference=1.0):
    value_db = 10.0 * np.log10(np.maximum(min_value, x))
    value_db -= 10.0 * np.log10(np.maximum(min_value, reference))
    return value_db


def get_spectrogram_corrector(shape=(1, 129, 375)):
    def corrector(x):
        if isinstance(x, xr.DataArray):
            x = x.squeeze("station")
            x = x.squeeze("network")
            x = x.squeeze("location")

        x = x[..., : shape[-1]]

        if isinstance(x, np.ndarray):
            x = x.reshape(shape)
        x = x.astype(np.float32)

        # we need to remove these coords because they are diverging between samples
        # they would introduce errors when batching afterwards
        if isinstance(x, xr.DataArray):
            del x["station"]
            del x["channel"]
            del x["time"]
            del x["network"]
            del x["location"]

        expected = shape
        if tuple(x.shape) != expected:
            raise RuntimeError(f"Expected shape {expected} but got {x.shape}")

        if pd.isnull(x).any():
            raise RuntimeError(f"Data is null")

        return x

    return corrector


def get_class_mapping(classes, class_rename=None, misc_class="misc", tags_name="tags"):
    def class_mapping(x):
        x_out = np.zeros((len(classes),))
        for e in x:
            for t in e[tags_name]:

                if class_rename is not None:
                    t = class_rename[t]
                if t not in classes:
                    continue
                x_out[classes[t]] = 1
        # 'misc' is our class for unknown events. if we have multiple labels and
        # misc is one of them, remove misc
        if x_out[classes[misc_class]] == 1 and x_out.sum() > 1:
            x_out[classes[misc_class]] = 0
        return xr.DataArray(
            x_out, dims=[tags_name], coords={tags_name: list(classes.keys())}
        )

    return class_mapping


from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(41760, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_taskgraphs(classes=None):
    if classes is None:
        classes = requests_from_file("./data/matterhorn_classes.jsonl")[0]

    # Create an instance of a seismic portal to load the data from arclink.ethz.ch
    seismic_inst = SeismicPortal(
        use_arclink={"url": "http://arclink.ethz.ch"},
        channel=["EHZ"],
        station=["MH36"],
        location=["A"],
        network=["1I"],
    )

    # we want to process the data as a spectrogram
    spectrogram_inst = Spectrogram(
        nfft=256,
        stride=8,
        dim="time",
    )

    """ We instantiate a simple CNN model taken from a 
[pytorch tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"""
    model = Net(len(classes))

    """Our model's task is to classifies the "classification target" (for example a 
time segment). It is not bounded to only use the classification target as input
it may use alternative sources. We call this the "classification scope".
In this case we use 
    - 90 seconds of data before the end of the classification target (no matter how 
    long the target actually is)
    - Three channels "EHZ",'EHE','EHN'
    - One station 'MH36
Since we need to know the end of the classification target, we create a function
that dynamically computes the scope based on the target request."""

    def classification_scope(request):
        rs = request["self"]
        start_time = rs["indexers"]["time"]["start"]  # unused but available
        stop_time = foreal.to_datetime(rs["indexers"]["time"]["stop"])

        classification_scope = {
            "time": {
                "start": stop_time - foreal.to_timedelta(90, "seconds"),
                "stop": stop_time,
            }
        }
        classification_scope["channel"] = ["EHZ", "EHE", "EHN"]
        classification_scope["station"] = ["MH36"]
        return classification_scope

    """The model can however only compute with a specific input size at once.
The classification scope might be larger (and is larger in our case). Therefore we slice
the classification scope into smaller segments, which can be processed by our model.
Then we need to take care of combining the model outputs for each slice afterwards.
Since the segment_slice and segment_stride do not depend on the classification target
we can just use dicts to describe them instead of a function. They can also be a 
function
    """
    # it should subdivide the input into 30 seconds slices ...
    segment_slice = {"time": foreal.to_timedelta(30, "seconds")}
    # ... containing 1 channel per slice ...
    segment_slice["channel"] = 1
    # ... and 1 station per slice ...
    segment_slice["station"] = 1

    # the stride/stepsize is also 30 seconds (no overlap between slices) ...
    segment_stride = {"time": foreal.to_timedelta(30, "seconds")}
    # ... every channel of `classification_scope["channel"]` will be stepped through ...
    segment_stride["channel"] = 1
    # ... and every station of `classification_scope["station"]`...
    segment_stride["station"] = 1

    # our "grid" of segments will be based on a reference time
    ref = {"time": pd.to_datetime("20200101")}
    # and all segments with an overlap with the classification scope will be used
    mode = {"time": "overlap"}

    modelwrapper_inst = ModelWrapperGenerator(
        model,
        batch_size=1,
        classification_scope=classification_scope,
        segment_slice=segment_slice,
        segment_stride=segment_stride,
        ref=ref,
        mode=mode,
    )

    dataset_name = "permalytix"
    """ Our model wrapper will produce many segments which we can use for training.
We can use these in modern machine learning frameworks if we collect them in a datasets.
"""
    persist_dir = Path(get_setting("persist_dir"))
    dataset = Dataset(
        persist_store=foreal.DirectoryStore(
            persist_dir / f"datasets/{dataset_name}_requests.foreal"
        )
    )

    """We will create for annotations way to load annotations for each segment
    """

    # annotations are a list of requests with a tag component
    annotations_filename = Path("./data") / "matterhorn_annotations.jsonl"
    # the schema defines which components the BoundingBoxAnnotation class should use
    # to compute an overlap between a request and the annotations
    schema = {
        "indexers": {"time": "datetimeInterval", "station": "string"},
    }
    bba = BoundingBoxAnnotation(annotations_filename, schema=schema)

    # we will syncronize the `dataset` and `annotation_dataset` later
    # therefore we do not add a persist_dir
    annotation_dataset = Dataset()

    """Additionally we introduce
    """
    hashpersist_dir = (
        persist_dir / f"datasets/{dataset_name}_dataset_hashpersist.foreal"
    )
    dataset_hashpersist = HashPersister(hashpersist_dir)
    hashpersist_dir = (
        persist_dir / f"datasets/{dataset_name}_annotationset_hashpersist.foreal"
    )
    annotationset_hashpersist = HashPersister(hashpersist_dir)

    d = {}
    with foreal.use_delayed():
        ######### SEISMIC WAVEFORMS #############
        seismic_node = seismic_inst(dask_key_name="seismic")
        ######## SPECTROGRAM   ###########
        spectrogram_node = spectrogram_inst(seismic_node, dask_key_name="spectrogram")
        spectrogram_db_node = foreal.it(to_db)(spectrogram_node, dask_key_name="to_db")

        # If there is any segment which has wrong dimensions for whatever reasons
        # we want to catch it here.
        spec_shape = (1, 129, 375)
        spectrogram_corrected_node = foreal.it(
            get_spectrogram_corrector(shape=spec_shape)
        )(spectrogram_db_node, dask_key_name="corrector")
        ##################################

        ######## DATASET #######
        x_dataset_hashpersist = dataset_hashpersist(
            spectrogram_corrected_node, dask_key_name="dataset_hashpersist"
        )
        x_dataset = dataset(x_dataset_hashpersist, dask_key_name="dataset")
        d["dataset"] = x_dataset
        ########################

        ######## ANNOTATIONS ############
        y_annotations = bba(dask_key_name="bba")
        # converting output of bba (request) to xarray DataArrays
        y_mapping = foreal.it(get_class_mapping(classes))(
            y_annotations, dask_key_name="mapping"
        )
        x_annotationset_hashpersist = annotationset_hashpersist(
            y_mapping, dask_key_name="annotationset_hashpersist"
        )
        y_dataset = annotation_dataset(
            x_annotationset_hashpersist, dask_key_name="annotation_dataset"
        )
        d["annotation_dataset"] = y_dataset
        #####################################

        ######## MODEL #######
        x_predictions = modelwrapper_inst(x_dataset, dask_key_name="modelwrapper")
        ######################

    d["default"] = x_predictions

    return d
