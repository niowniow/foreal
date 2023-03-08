# %%[markdown]
"""
# Machine Learning example
"""
# %%
import copy
import os
import re
from argparse import ArgumentError, Namespace
from inspect import classify_class_attrs

import dask
import joblib
import numpy as np
import pandas as pd
import torch
from ignite.metrics import Accuracy
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import foreal
from foreal.apps.annotations import (
    requests_from_file,
    requests_to_file,
    to_formatted_json,
)
from foreal.apps.models import ModelWrapperGenerator, get_segments
from foreal.config import get_global_config, set_setting
from foreal.convenience import dict_update, to_datetime
from foreal.core import configuration
from foreal.core.datasets import ForealDataset, ForealDatasetHash

from .taskgraphs.classification import get_taskgraphs

# %%[markdown]
"""
## Fail modes
We can define several modes how error should be treated. If we know that e.g. certain
files are not available and will cause an error, we can set the mode to `warning` or
`ignore`.
"""
# %%
set_setting("fail_mode", "warning")
# set_setting("fail_mode", "ignore")
# set_setting("fail_mode", "fail")

# set dask to single-threaded. We'll be using pytorch multithreading
dask.config.set(scheduler="single-threaded")

# in this example we just set a fixed directory as persist dir
# you can however also change it to a config file
set_setting("persist_dir", "./tmp")
os.makedirs("./tmp", exist_ok=True)


# %%[markdown]


# %%[markdown]
"""
## Loading annotations
All of our annotations are stored within a jsonl file, one annotation per line.
We will only consider a subset of all labels in our annotation set.
Also, there are many annotations in the file. We reduce it to a few to minimize loading 
time since in this example, we will load the data directly from the server without storing it
"""
# %%
# consider only a subset
classes = requests_from_file("./data/matterhorn_classes.jsonl")[0]
classes_subset = ["misc", "mountaineer"]
print(f"Using the following subset {classes_subset} of all classes {classes}")

annotation_requests = requests_from_file("./data/matterhorn_annotations.jsonl")

# keep only misc and mountaineers
annotation_requests = [
    a for a in annotation_requests if "misc" in a["tags"] or "mountaineer" in a["tags"]
]

# keep only a few misc and a few mountainer annotations
annotation_requests_small = []
max_annotations_per_tag = 5
counter = {}
for c in annotation_requests:
    for tag in c["tags"]:
        if counter.get(tag, 0) > max_annotations_per_tag:
            continue
        counter[tag] = counter.get(tag, 0) + 1
        annotation_requests_small += [c]
    if all([counter[tag] > max_annotations_per_tag for tag in counter]):
        break

# %%[markdown]
"""
foreal uses numpy/xarray underneath. These framework cannot handle timezones properly
Therefore the foReal's internal time representation is utc and tz-naive (!)
and we need to convert all timestamps to the internal foreal format.
"""
# %%
# convert all timestamps to utc tz-naive (i.e. convert utc and then remove tz info because xarray/numpy can't handle it)
for c in annotation_requests_small:
    c["indexers"]["time"]["start"] = to_datetime(
        c["indexers"]["time"]["start"]
    ).isoformat()
    c["indexers"]["time"]["stop"] = to_datetime(
        c["indexers"]["time"]["stop"]
    ).isoformat()


print(annotation_requests_small)
# %%[markdown]
"""
Our classifier is trained to classify a segment, i.e.:
Is a event happening during this segment or not?
we call this segment/timestamp a classification target
1. we subdivide the whole total time period into a grid with a step size S
2. each segment of that grid is part of our train/test set
3. for training/testing we subsequentially remove entries to reduce the dataset scope.
   We keep all segments which overlap with annotated events and remove the rest.
   In a classification scenario we would classify each classification target
"""
# %%
# create a grid with a common reference time and overlap it with our annotation set
ref = {"time": pd.to_datetime("20200101")}
print(ref)
mode = {"time": "overlap"}
segment_slice = {"time": foreal.to_timedelta(30, "seconds")}
segment_stride = {"time": foreal.to_timedelta(30, "seconds")}
classification_segments = []
for c in annotation_requests_small:
    non_sliced_segment = {
        "time": {
            "start": c["indexers"]["time"]["start"],
            "stop": c["indexers"]["time"]["stop"],
        }
    }
    segments = get_segments(
        non_sliced_segment,
        segment_slice,
        segment_stride,
        ref=ref,
        mode=mode,
        timestamps_as_strings=True,
        minimal_number_of_segments=1,
    )
    classification_segments += segments.tolist()


# transform in proper foreal requests
classification_targets = np.array([{"indexers": c} for c in classification_segments])
print(classification_targets)

dataset_name = "permalytix"

# here we load the actuall task graph
d = get_taskgraphs(classes=classes)

predictions_taskgraph = d["default"]

datasets = {}
subset = "all"

print("initializing dataset")
# 1. record or load a set of requests into the dataset using the name of the subset
#    If the subset is in dataset.requests is has already been loaded from disk
dataset = foreal.extract_node_from_graph(predictions_taskgraph, "dataset")
if not dataset.is_persisted():
    print("Recording a set of requests into the dataset:", "dataset")
    for r in tqdm(classification_targets):
        r = copy.deepcopy(r)
        dict_update(
            r,
            {
                "config": {
                    "subset": subset,
                    # we tell the dataset to record this call
                    "keys": {
                        "dataset": {"record": True},
                    },
                },
            },
        )
        _ = configuration(d["default"], r, optimize_graph=False)

    dataset.persist()
else:
    print("Dataset already loaded from disk")

print("initializing annotation set")
# instead of recording a set into the annotation_dataset
# we just copy set from the dataset
annotation_dataset = foreal.extract_node_from_graph(
    d["annotation_dataset"], "annotation_dataset"
)
# making sure annotation set and data set have the same requests
annotation_dataset.requests = dataset.requests

# # 3. pre-loading the whole persister
print("pre-loading the dataset")


request_base = {
    "config": {
        "subset": subset,
    },
}

dataset_node = d["dataset"]
dataset_hashpersist_instance = foreal.extract_node_from_graph(
    predictions_taskgraph, "dataset_hashpersist"
)
fdata_set = ForealDatasetHash(
    dataset,
    dataset_node,
    persister=dataset_hashpersist_instance,
    request_base=request_base,
    subset=subset,
)
print("going to preload")

fdata_set.preload(batch_size=4, num_workers=0)
print(len(fdata_set))
fdata_set.mask_invalid()
print(len(fdata_set))

annotation_dataset_hashpersist_instance = foreal.extract_node_from_graph(
    d["annotation_dataset"], "annotationset_hashpersist"
)
fannotation_set = ForealDatasetHash(
    annotation_dataset,
    d["annotation_dataset"],
    persister=annotation_dataset_hashpersist_instance,
    request_base=request_base,
    subset=subset,
)
fannotation_set.preload(batch_size=16, num_workers=0)
print(len(fannotation_set))
fannotation_set.mask_invalid()
print(len(fannotation_set))
train_dataset = ForealDatasetHash.join([fdata_set, fannotation_set])
test_dataset = ForealDatasetHash.join([fdata_set, fannotation_set])

# let's check what our dataset item actually look like
# they should be a tuple of xarray data and annotation map
print(train_dataset[0])
print(train_dataset[1])


# the transform is used during dataset loading
# it can be adapted based on the data augmenation needed.
def transform(x):
    x = x.values
    x = x[0]
    x = torch.Tensor(x)
    return x


# We manually add the transformation to the sets
train_dataset.transforms[0] = transform
train_dataset.transforms[1] = transform
test_dataset.transforms[0] = transform
test_dataset.transforms[1] = transform

train_dataset.indices = train_dataset.indices[: len(train_dataset) * 3 // 4]
test_dataset.indices = test_dataset.indices[len(test_dataset) * 3 // 4 :]

## generate some statistics


def stats(dataset, classes_subset, annotation_dataset_index=1):
    dataset[0, annotation_dataset_index]  # init

    values = np.array([dataset[i, 1].numpy() for i in range(len(dataset))]).astype(bool)
    print("shape used", values.shape)

    print(
        "dataset Annotation distribution per class",
        dict(zip(classes_subset, values.sum(axis=0))),
    )
    print("Annotation distribution no-class", (values.sum(axis=1) == 0).sum())
    print("Annotation multiclass", np.bincount(values.sum(axis=1)))
    # print('Annotation multiclass',dataset.requests[values.sum(axis=1)>1])
    print("Total annotated dataset length", len(dataset))


print("\n####################")
print("train dataset statistics")
stats(train_dataset, classes)

print("\n####################")
print("test dataset statistics")
stats(test_dataset, classes)

print("start training")
# now we can use the datasets for training the model
model = foreal.extract_node_from_graph(predictions_taskgraph, "modelwrapper").model

batch_size = 32
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)

testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

print("Finished Training")


acc = Accuracy(is_multilabel=True)
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)

        y_pred = outputs.sigmoid().round()
        acc.update((y_pred, labels))


acc_value = acc.compute()
print(f"Accuracy of the network on the 10000 test images: {100 * acc_value} %")
