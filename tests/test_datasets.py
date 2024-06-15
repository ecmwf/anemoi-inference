# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from functools import lru_cache

import pytest
from anemoi.datasets import open_dataset
from anemoi.datasets.data.stores import zarr_lookup

from anemoi.inference.checkpoint.metadata import Metadata

zarr_lookup = lru_cache(maxsize=None)(zarr_lookup)


def missing(*names):
    for name in names:
        if zarr_lookup(name, fail=False) is None:
            return True
    return False


def any_missing(dataset):
    if isinstance(dataset, dict):
        if "dataset" in dataset:
            if missing(dataset["dataset"]):
                return True

        if "datasets" in dataset:
            for v in dataset["datasets"]:
                if isinstance(v, str):
                    if missing(v):
                        return True
                elif any_missing(v):
                    return True

        for k, v in dataset.items():
            if any_missing(v):
                return True

        return False

    if isinstance(dataset, (list, tuple)):
        for v in dataset:
            if any_missing(v):
                return True

        return False

    return False


def standard_test(dataset):

    if any_missing(dataset):
        pytest.skip(f"Dataset {dataset} not available")

    ds = open_dataset(**dataset)
    md = Metadata.from_metadata({"version": "1.0.0", "dataset": ds.metadata()})
    md.dump()

    md.digraph()

    return ds, md


def test_cerra():
    standard_test({"dataset": "cerra-rr-an-oper-0001-mars-5p5km-1984-2008-6h-v1-smhi"})


def test_cutout():
    standard_test(
        {
            "adjust": "all",
            "cutout": [
                {"dataset": "cerra-rr-an-oper-0001-mars-5p5km-1984-2008-6h-v1-smhi", "thinning": 25},
                {"dataset": "aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6"},
            ],
        }
    )


def test_grids():
    standard_test(
        {
            "adjust": "all",
            "grids": [
                {"dataset": "cerra-rr-an-oper-0001-mars-5p5km-1984-2008-6h-v1-smhi", "thinning": 25},
                {"dataset": "aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6"},
            ],
        }
    )


def test_rename():
    standard_test(
        {
            "dataset": "aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6",
            "rename": {"2t": "t2m"},
        }
    )


def test_drop():
    standard_test(
        {
            "dataset": "aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6",
            "drop": {"2t"},
        }
    )


def test_select():
    standard_test(
        {
            "dataset": "aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6",
            "select": {"2t"},
        }
    )


def test_subset():
    standard_test(
        {
            "dataset": "aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6",
            "start": "2010-01-01",
            "end": "2010-01-02",
            "frequency": "6h",
        }
    )


def test_ensemble():
    standard_test(
        {
            "datasets": [
                "aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6",
                "aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6",
            ]
        }
    )


def test_join():
    standard_test(
        {
            "datasets": [
                {"dataset": "aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6", "select": {"2t"}},
                {"dataset": "aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6", "select": {"2d"}},
            ],
        }
    )


def test_concat():
    standard_test(
        {
            "datasets": [
                {"dataset": "aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6", "end": 2010},
                {"dataset": "aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6", "start": 2011},
            ],
        }
    )


def test_statistics():
    standard_test(
        {
            "dataset": "aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6",
            "statistics": {"dataset": "aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6"},
        }
    )


if __name__ == "__main__":
    test_drop()
