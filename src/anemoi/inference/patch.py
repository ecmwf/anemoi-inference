# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
from contextlib import contextmanager
from functools import cached_property

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

    def patch_metadata(self, supporting_arrays, root, force=False):
        dataset = self._metadata["dataset"]

        if (
            "variable_metadata" not in dataset
            or "supporting_arrays_paths" not in dataset
            or "sources" not in dataset
            or not not self._supporting_arrays
            or force
        ):
            self._patch_variable_metadata()
            self._supporting_arrays = self._patch_supporting_arrays(supporting_arrays, root)

        return self._metadata, self._supporting_arrays

    def _patch_variable_metadata(self):

        try:
            return self._patch_variable_metadata_open_dataset_1()
        except Exception:
            LOG.exception("_patch_variable_metadata_open_dataset_1 failed")

        try:
            return self._patch_variable_metadata_open_dataset_2()
        except Exception:
            LOG.exception("_patch_variable_metadata_open_dataset_2 failed")

    @cached_property
    def _from_zarr(self):
        """We assume that the datasets are reachable via the content of
        ~/.config/anemoi/settings.toml
        """
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
        return ds.metadata(), ds.supporting_arrays()

    def _patch_variable_metadata_open_dataset_1(self):
        """Try to open the dataset(s) and re-fetch metadata.
        In the checkpoint we keep track of the arguments used to open the dataset.
        """

        # First attempt, try to open the dataset

        dataset = self._metadata["dataset"]
        metadata, _ = self._from_zarr

        # Update the metadata
        for k, v in metadata.items():
            if k not in dataset:
                LOG.info("Updating metadata key `%s` with value `%s`", k, v)
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

    def _patch_supporting_arrays(self, supporting_arrays, root):

        metadata, supporting_arrays = self._from_zarr

        # assert False, metadata['sources']

        LOG.info("Supporting arrays: '%s'", metadata["supporting_arrays"])
        LOG.info("Supporting arrays: '%s'", supporting_arrays)

        supporting_arrays_paths = {
            key: dict(
                path=f"{root}/{key}.numpy",
                shape=value.shape,
                dtype=str(value.dtype),
            )
            for key, value in supporting_arrays.items()
        }

        for k, v in supporting_arrays_paths.items():
            LOG.info("Saving supporting array `%s` to %s (shape=%s, dtype=%s)", k, v["path"], v["shape"], v["dtype"])

        self._metadata["supporting_arrays_paths"] = supporting_arrays_paths

        return supporting_arrays
