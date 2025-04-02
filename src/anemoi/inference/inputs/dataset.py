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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from earthkit.data.utils.dates import to_datetime

from anemoi.inference.context import Context
from anemoi.inference.input import Input
from anemoi.inference.types import Date
from anemoi.inference.types import FloatArray
from anemoi.inference.types import State

from . import input_registry

LOG = logging.getLogger(__name__)


class DatasetInput(Input):
    """Handles `anemoi-datasets` dataset as input."""

    def __init__(self, context: Context, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        """Initialize the DatasetInput.

        Parameters
        ----------
        context : Any
            The context in which the input is used.
        args : Tuple[Any, ...]
            Arguments for the dataset.
        kwargs : Dict[str, Any]
            Keyword arguments for the dataset.
        """
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
    def ds(self) -> Any:
        """Return the dataset."""
        from anemoi.datasets import open_dataset

        return open_dataset(*self.args, **self.kwargs)

    @cached_property
    def latitudes(self) -> FloatArray:
        """Return the latitudes."""
        return self.ds.latitudes

    @cached_property
    def longitudes(self) -> FloatArray:
        """Return the longitudes."""
        return self.ds.longitudes

    def __repr__(self) -> str:
        """Return a string representation of the DatasetInput."""
        return f"DatasetInput({self.args}, {self.kwargs})"

    def create_state(
        self, *, date: Optional[Date], variables: Optional[List[str]] = None, initial: bool = True
    ) -> State:
        """Create the input state for the given date.

        Parameters
        ----------
        date : Optional[Any]
            The date for which to create the input state.
        variables : Optional[List[str]]
            The list of variables to include in the input state.
        initial : bool
            Whether the state is the initial state, in which case date expands to a list of dates
            according to the model's input time window lag.

        Returns
        -------
        State
            The created input state.
        """

        # prepare request to zarr store
        date = to_datetime(date)
        latitudes = self.ds.latitudes[self.grid_indices]
        longitudes = self.ds.longitudes[self.grid_indices]
        variables = self.checkpoint_variables if variables is None else variables
        variables_indexer = [self.ds.variables.index(v) for v in variables]
        dates = [date + lag for lag in self.checkpoint.lagged] if initial else [date]
        dates_indexer = self._dates_idx(dates)

        # get data from zarr store
        data = self.ds[dates_indexer]
        data = data[:, variables_indexer]
        if data.shape[2] != 1:
            raise ValueError(f"Ensemble data not supported, got {data.shape[2]} members")
        data = np.squeeze(data, axis=2)  # squeeze ensemble dimension

        # create the state
        state = {"date": date, "latitudes": latitudes, "longitudes": longitudes, "fields": {}}
        for i, variable in enumerate(variables):
            state["fields"][variable] = data[:, i]
            if self.context.trace:
                self.context.trace.from_input(variable, self)

        return state

    def _dates_idx(self, dates: List[Date]) -> slice:
        """Return the slice corresponding to provided dates within the dataset.

        Parameters
        ----------
        dates : List[Date]
            Sorted list of dates to find in the dataset.

        Returns
        -------
        slice
            Slice object representing indices of the dates.

        Raises
        ------
        ValueError
            If a date is not found in the dataset, or if dates do not form a regular interval.
        """
        dataset_dates = self.ds.dates

        # Map dates to their indices using np.searchsorted (efficient for sorted arrays)
        dates_idx = np.searchsorted(dataset_dates, dates)
        missing_dates = [d for idx, d in zip(dates_idx, dates) if idx == len(dataset_dates)]
        if missing_dates != []:
            raise ValueError(f"Dates not found in dataset: {missing_dates}")

        if len(dates_idx) == 1:
            return slice(dates_idx[0], dates_idx[0] + 1)

        steps = np.diff(dates_idx)
        step = steps[0]

        if not np.all(steps == step):
            raise ValueError(f"Dates do not have regular intervals: indices {dates_idx}")

        return slice(dates_idx[0], dates_idx[-1] + 1, step)


@input_registry.register("dataset")
class DatasetInputArgsKwargs(DatasetInput):
    """Handles `anemoi-datasets` dataset as input."""

    trace_name = "dataset/provided"

    def __init__(self, context: Context, /, *args: Any, use_original_paths: bool = True, **kwargs: Any) -> None:
        """Initialize the DatasetInputArgsKwargs.

        Parameters
        ----------
        context : Context
            The context in which the input is used.
        use_original_paths : bool
            Whether to use original paths.
        """
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
    """Handles `anemoi-datasets` dataset as input."""

    def __init__(self, context: Context, /, name: str, use_original_paths: bool = True, **kwargs: Any) -> None:
        """Initialize the DataloaderInput.

        Parameters
        ----------
        context : Any
            The context in which the input is used.
        name : str
            The name of the dataloader.
        use_original_paths : bool
            Whether to use original paths.
        """
        args, kwargs = context.checkpoint.open_dataset_args_kwargs(
            use_original_paths=use_original_paths,
            from_dataloader=name,
        )

        super().__init__(context, args, kwargs)


@input_registry.register("test")
class TestInput(DataloaderInput):
    """Handles `anemoi-datasets` dataset as input."""

    trace_name = "dataset/test"

    def __init__(self, context: Context, /, use_original_paths: bool = True, **kwargs: Any) -> None:
        """Initialize the TestInput.

        Parameters
        ----------
        context : Any
            The context in which the input is used.
        use_original_paths : bool
            Whether to use original paths.
        """
        super().__init__(
            context,
            name="test",
            use_original_paths=use_original_paths,
            **kwargs,
        )


@input_registry.register("validation")
class ValidationInput(DataloaderInput):
    """Handles `anemoi-datasets` dataset as input."""

    trace_name = "dataset/validation"

    def __init__(self, context: Context, /, use_original_paths: bool = True, **kwargs: Any) -> None:
        super().__init__(
            context,
            name="validation",
            use_original_paths=use_original_paths,
            **kwargs,
        )


@input_registry.register("training")
class TrainingInput(DataloaderInput):
    """Handles `anemoi-datasets` dataset as input."""

    trace_name = "dataset/training"

    def __init__(self, context: Context, /, use_original_paths: bool = True, **kwargs: Any) -> None:
        super().__init__(
            context,
            name="training",
            use_original_paths=use_original_paths,
            **kwargs,
        )
