# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import json
import logging
from functools import cached_property

import numpy as np
from earthkit.data.utils.dates import to_datetime

from ..input import Input
from . import input_registry

LOG = logging.getLogger(__name__)


class DatasetInput(Input):
    """
    Handles `anemoi-datasets` dataset as input
    """

    def __init__(self, context, args, kwargs):
        super().__init__(context)

        grid_indices = kwargs.pop("grid_indices", None)

        self.args, self.kwargs = args, kwargs
        if context.verbosity > 0:
            LOG.info(
                "Opening dataset with\nargs=%s\nkwargs=%s", json.dumps(args, indent=4), json.dumps(kwargs, indent=4)
            )

        if grid_indices is None and "grid_indices" in context.checkpoint._supporting_arrays:
            grid_indices = context.checkpoint.load_supporting_array("grid_indices")
            if context.verbosity > 0:
                LOG.info(
                    "Loading supporting array `grid_indices` from checkpoint, \
                    the input grid will be reduced accordingly."
                )

        self.grid_indices = slice(None) if grid_indices is None else grid_indices

    @cached_property
    def ds(self):
        from anemoi.datasets import open_dataset

        return open_dataset(*self.args, **self.kwargs)

    def __repr__(self):
        return f"DatasetInput({self.args}, {self.kwargs})"

    def create_input_state(self, *, date=None):
        if date is None:
            raise ValueError("`date` must be provided")

        date = to_datetime(date)
        latitudes = self.ds.latitudes
        longitudes = self.ds.longitudes

        input_state = dict(
            date=date,
            latitudes=latitudes[self.grid_indices],
            longitudes=longitudes[self.grid_indices],
            fields=dict(),
        )

        fields = input_state["fields"]

        date = np.datetime64(date)
        data = self._load_dates([date + np.timedelta64(h) for h in self.checkpoint.lagged])

        if data.shape[2] != 1:
            raise ValueError(f"Ensemble data not supported, got {data.shape[2]} members")

        requested_variables = set(self.input_variables())
        for i, variable in enumerate(self.ds.variables):
            if variable not in requested_variables:
                continue
            # Squeeze the data to remove the ensemble dimension
            values = np.squeeze(data[:, i], axis=1)
            fields[variable] = values[:, self.grid_indices]

        return input_state

    def load_forcings(self, *, variables, dates):
        data = self._load_dates(dates)  # (date, variables, ensemble, values)

        requested_variables = np.array([self.ds.name_to_index[v] for v in variables])
        data = data[:, requested_variables]
        # Squeeze the data to remove the ensemble dimension
        data = np.squeeze(data, axis=2)
        # Reorder the dimensions to (variable, date, values)
        data = np.swapaxes(data, 0, 1)
        # apply reduction to `grid_indices`
        data = data[..., self.grid_indices]
        return data

    def _load_dates(self, dates):

        # TODO: use the fact that the dates are sorted

        dataset_dates = self.ds.dates

        idx = []
        for d in dates:
            (i,) = np.where(dataset_dates == d)
            if len(i) == 0:
                raise ValueError(
                    f"Date {d} not found in the dataset, available dates: {dataset_dates[0]}...{dataset_dates[-1]} by {self.ds.frequency}"
                )
            assert len(i) == 1, f"Multiple dates found for {d}"
            idx.append(int(i[0]))

        if len(idx) == 1:
            s = slice(idx[0], idx[0] + 1)
        else:
            diff = idx[1] - idx[0]
            if not all(i == diff for i in np.diff(idx)):
                # TODO: remove that restriction
                raise ValueError("Dates do not have the same frequency")
            s = slice(idx[0], idx[-1] + 1, diff)

        return self.ds[s]


@input_registry.register("dataset")
class DatasetInputArgsKwargs(DatasetInput):
    """Handles `anemoi-datasets` dataset as input"""

    def __init__(self, context, /, *args, use_original_paths=True, **kwargs):
        if not args and not kwargs:
            args, kwargs = context.checkpoint.open_dataset_args_kwargs(use_original_paths=use_original_paths)

            # TODO: remove start/end from the arguments

            LOG.warning("No arguments provided to open_dataset, using the default arguments:")

            cmd = "open_dataset("
            for arg in args:
                cmd += f"{arg}, "
            for key, value in kwargs.items():
                cmd += f"{key}={value}, "

            LOG.warning("%s", cmd)
        super().__init__(context, args, kwargs)


class DataloaderInput(DatasetInput):
    """Handles `anemoi-datasets` dataset as input"""

    def __init__(self, context, /, name, use_original_paths=True, **kwargs):

        args, kwargs = context.checkpoint.open_dataset_args_kwargs(
            use_original_paths=use_original_paths,
            from_dataloader=name,
        )

        super().__init__(context, args, kwargs)


@input_registry.register("test")
class TestInput(DataloaderInput):
    """Handles `anemoi-datasets` dataset as input"""

    def __init__(self, context, /, use_original_paths=True, **kwargs):
        super().__init__(context, name="test", use_original_paths=use_original_paths, **kwargs)


@input_registry.register("validation")
class ValidationInput(DataloaderInput):
    """Handles `anemoi-datasets` dataset as input"""

    def __init__(self, context, /, use_original_paths=True, **kwargs):
        super().__init__(context, name="validation", use_original_paths=use_original_paths, **kwargs)


@input_registry.register("training")
class TrainingInput(DataloaderInput):
    """Handles `anemoi-datasets` dataset as input"""

    def __init__(self, context, /, use_original_paths=True, **kwargs):
        super().__init__(context, name="training", use_original_paths=use_original_paths, **kwargs)
