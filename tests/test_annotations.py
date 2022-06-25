import numpy as np

import foreal
from foreal.apps import annotations
from foreal.apps.annotations import BoundingBoxAnnotation


def compare_requests(r1, r2, keys=None):
    def cp(r1, r2, keys):
        if keys is None:
            keys_r1 = r1.keys()
        else:
            keys_r1 = keys
        for key in keys_r1:
            if key in r2:
                if isinstance(r1[key], dict) and isinstance(r1[key], dict):
                    if not compare_requests(r1[key], r2[key]):
                        return False
                else:
                    if r1[key] != r2[key]:
                        return False
            else:
                return False

        return True

    return cp(r1, r2, keys) and cp(r2, r1, keys)


def is_in(ann, data, keys):
    for a in ann:
        cond = False
        for d in data:
            if compare_requests(d, a, keys):
                cond = True
        assert cond
    return True


def test_annotations():
    schema = {"indexers": {"time": "datetimeInterval"}, "station": "string"}
    annotations = np.array(
        [
            {
                "indexers": {"time": {"start": "2016-01-14", "stop": "2016-01-17"}},
                "targets": {"middle": True},
                "station": "MH36",
            },
            {
                "indexers": {"time": {"start": "2016-01-15", "stop": "2016-01-16"}},
                "targets": {"middle": True},
                "station": "MH36",
            },
            {
                "indexers": {"time": {"start": "2016-01-18", "stop": "2016-01-19"}},
                "targets": {"after": True},
                "station": "MH36",
            },
            {
                "indexers": {"time": {"start": "2016-01-12", "stop": "2016-01-13"}},
                "targets": {"before": True},
                "station": "MH36",
            },
        ]
    )
    bba = BoundingBoxAnnotation(annotations, schema=schema)
    with foreal.use_delayed():
        x_annotation = bba(dask_key_name="bba")

    requests = [
        {
            "indexers": {"time": {"start": "2016-01-01", "stop": "2016-01-02"}},
            "station": "MH36",
        },  # fully before
        {
            "indexers": {"time": {"start": "2016-01-30", "stop": "2016-01-31"}},
            "station": "MH36",
        },  # fully after
        {
            "indexers": {"time": {"start": "2016-01-01", "stop": "2016-01-31"}},
            "station": "MH36",
        },  # start before / end after
        {
            "indexers": {"time": {"start": "2016-01-01", "stop": "2016-01-15"}},
            "station": "MH36",
        },  # start before / end in
        {
            "indexers": {"time": {"start": "2016-01-15", "stop": "2016-01-31"}},
            "station": "MH36",
        },  # start in / end after
        {
            "indexers": {"time": {"start": "2016-01-15", "stop": "2016-01-16"}},
            "station": "MH36",
        },
    ]  # start in / end in

    request = {bba.dask_key_name: {"requests": requests}}

    data = foreal.compute(x_annotation, request)

    # fully before
    assert len(data[0]) == 0 and not data[0]

    # fully after
    assert len(data[1]) == 0 and not data[1]

    # start before / end after
    # must include annotations 0, 1, 2, 3
    ann = annotations[[0, 1, 2, 3]]
    assert is_in(ann, data[2], ["indexers", "station"])
    assert is_in(data[2], ann, ["indexers", "station"])

    # start before / end in
    # must include annotations 0, 1, 3
    ann = annotations[[0, 1, 3]]
    assert is_in(ann, data[3], ["indexers", "station"])
    assert is_in(data[3], ann, ["indexers", "station"])

    # start in / end after
    # must include annotations 0, 1, 2
    ann = annotations[[0, 1, 2]]
    assert is_in(ann, data[4], ["indexers", "station"])
    assert is_in(data[4], ann, ["indexers", "station"])

    # start in / end after
    # must include annotations 0, 1
    ann = annotations[[0, 1]]
    assert is_in(ann, data[5], ["indexers", "station"])
    assert is_in(data[5], ann, ["indexers", "station"])
