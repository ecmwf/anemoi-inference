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

import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.types import Date
from anemoi.inference.types import FloatArray
from anemoi.inference.types import State

from ..input import Input
from . import input_registry

LOG = logging.getLogger(__name__)


class DatasetInput(Input):
    """Handles `anemoi-datasets` dataset as input."""

    def __init__(self, context: Context, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
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

    def create_input_state(self, *, date: Date | None = None) -> State:
        """Create the input state for the given date.

        Parameters
        ----------
        date : Optional[Any]
            The date for which to create the input state.

        Returns
        -------
        Dict[str, Any]
            The created input state.
        """
        if date is None:
            raise ValueError("`date` must be provided")

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
        dates = [date + np.timedelta64(h) for h in self.checkpoint.lagged]

        data = self._load_dates(dates)

        if data.shape[2] != 1:
            raise ValueError(f"Ensemble data not supported, got {data.shape[2]} members")

        requested_variables = set(self.input_variables())
        for i, variable in enumerate(self.ds.variables):
            if variable not in requested_variables:
                continue
            # Squeeze the data to remove the ensemble dimension
            values = np.squeeze(data[:, i], axis=1)
            fields[variable] = values[:, self.grid_indices]

            if self.context.trace:
                self.context.trace.from_input(variable, self)

        return input_state

    def load_forcings_state(self, *, variables: list[str], dates: list[Date], current_state: State) -> State:
        """Load the forcings state for the given variables and dates.

        Parameters
        ----------
        variables : List[str]
            List of variables to load.
        dates : List[Any]
            List of dates for which to load the forcings.
        current_state : State
            The current state of the input.

        Returns
        -------
        State
            The loaded forcings state.
        """
        data = self._load_dates(dates)  # (date, variables, ensemble, values)

        requested_variables = np.array([self.ds.name_to_index[v] for v in variables])
        data = data[:, requested_variables]
        # Squeeze the data to remove the ensemble dimension
        data = np.squeeze(data, axis=2)
        # Reorder the dimensions to (variable, date, values)
        data = np.swapaxes(data, 0, 1)

        # apply reduction to `grid_indices`
        data = data[..., self.grid_indices]

        fields = {v: data[i] for i, v in enumerate(variables)}

        return dict(
            fields=fields,
            dates=dates,
            latitudes=self.latitudes,
            longitudes=self.longitudes,
        )

    def _load_dates(self, dates: list[Date]) -> Any:
        """Load the data for the given dates.

        Parameters
        ----------
        dates : List[Any]
            List of dates for which to load the data.

        Returns
        -------
        Any
            The loaded data.
        """
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
