# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
from contextlib import contextmanager

from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta

LOG = logging.getLogger(__name__)


@contextmanager
def patch_function(target, attribute, replacement):
    original = getattr(target, attribute)
    setattr(target, attribute, replacement)
    try:
        yield
    finally:
        setattr(target, attribute, original)


class PatchMixin:

    # `self` is a `Metadata` object

    def patch_metadata(self):
        dataset = self._metadata["dataset"]

        if "variable_metadata" not in dataset:
            self._patch_variable_metadata()

        return self._metadata

    def _patch_variable_metadata(self):

        try:
            return self._patch_variable_metadata_open_dataset_1()
        except Exception:
            LOG.exception("_patch_variable_metadata_open_dataset_1 failed")

        try:
            return self._patch_variable_metadata_open_dataset_2()
        except Exception:
            LOG.exception("_patch_variable_metadata_open_dataset_2 failed")

    def _patch_variable_metadata_open_dataset_1(self):
        """Try to open the dataset(s) and re-fetch metadata.
        In the checkpoint we keep track of the arguments used to open the dataset.
        We assume that the datasets are reachable via the content of
        ~/.config/anemoi/settings.toml
        """

        # First attempt, try to open the dataset
        from anemoi.datasets import open_dataset

        dataset = self._metadata["dataset"]
        arguments = dataset["arguments"]

        def _(x):
            if isinstance(x, dict):
                return {k: _(v) for k, v in x.items()}
            if isinstance(x, list):
                return [_(v) for v in x]

            if isinstance(x, str):
                if x.endswith(".zarr"):
                    x = os.path.basename(x)
                    x = os.path.splitext(x)[0]

            return x

        args, kwargs = _(arguments["args"]), _(arguments["kwargs"])

        LOG.info(f"Opening dataset with args {args} and kwargs {kwargs}")
        ds = open_dataset(*args, **kwargs)

        # Update the metadata
        for k, v in ds.metadata().items():
            if k not in dataset:
                dataset[k] = v

    def _patch_variable_metadata_open_dataset_2(self):
        """That version fetches the metadata from the catalogue."""

        import anemoi.datasets.data.stores
        import numpy as np
        import zarr
        from anemoi.registry import Dataset

        start_date = as_datetime(self._metadata["dataset"]["start_date"])
        end_date = as_datetime(self._metadata["dataset"]["end_date"])
        frequency = to_timedelta(self._metadata["dataset"]["frequency"])
        dates = []
        while start_date <= end_date:
            dates.append(start_date)
            start_date += frequency

        def _open_zarr(name):
            nonlocal dates
            name = os.path.basename(name)
            name = os.path.splitext(name)[0]
            entry = Dataset(name)

            root = zarr.group()
            root.attrs.update(entry.record["metadata"])
            values = 10
            variables = 10

            data = np.zeros(shape=(2, variables, 1, values))
            root.create_dataset(
                "data",
                # data=data,
                dtype=data.dtype,
                shape=(2, variables, 1, values),
                # compressor=None,
            )
            dates = np.array(dates, dtype="datetime64")
            root.create_dataset(
                "dates",
                data=dates,
                compressor=None,
            )
            root.create_dataset(
                "latitudes",
                shape=(values,),
                # compressor=None,
            )
            root.create_dataset(
                "longitudes",
                shape=(values,),
                # compressor=None,
            )

            return root

        with patch_function(anemoi.datasets.data.stores, "open_zarr", _open_zarr):
            self._patch_variable_metadata_open_dataset_1()