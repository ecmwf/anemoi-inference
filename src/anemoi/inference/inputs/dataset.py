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

from anemoi.inference.context import Context
from anemoi.inference.types import Date
from anemoi.inference.types import FloatArray
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State

from ..input import Input
from . import input_registry

LOG = logging.getLogger(__name__)


class DatasetInput(Input):
    """Handles `anemoi-datasets` dataset as input."""

    def __init__(
        self,
        context: Context,
        *,
        open_dataset_args: Tuple[Any, ...],
        open_dataset_kwargs: Dict[str, Any],
        grid_indices: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DatasetInput.

        Parameters
        ----------
        context : Any
            The context in which the input is used.
        open_dataset_args : Tuple[Any, ...]
            Arguments for the dataset.
        open_dataset_kwargs : Dict[str, Any]
            Keyword arguments for the dataset.
        grid_indices : Optional[Any]
            Indices to reduce the input grid. If None, the full grid is used.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(context, **kwargs)

        self.open_dataset_args = open_dataset_args
        self.open_dataset_kwargs = open_dataset_kwargs

        if context.verbosity > 0:
            LOG.info(
                "Opening dataset with\nargs=%s\nkwargs=%s",
                json.dumps(open_dataset_args, indent=4),
                json.dumps(
                    open_dataset_kwargs,
                    indent=4,
                ),
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

        dataset = open_dataset(*self.open_dataset_args, **self.open_dataset_kwargs)
        if self.variables is not None:
            dataset = open_dataset(dataset, select=self.variables)

        return dataset

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
        return f"DatasetInput({self.open_dataset_args}, {self.open_dataset_kwargs})"

    def create_input_state(self, *, date: Optional[Date] = None) -> State:
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

    def load_forcings_state(self, *, dates: List[Date], current_state: State) -> State:
        """Load the forcings state for the given variables and dates.

        Parameters
        ----------
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

        requested_variables = np.array([self.ds.name_to_index[v] for v in self.variables])
        data = data[:, requested_variables]
        # Squeeze the data to remove the ensemble dimension
        data = np.squeeze(data, axis=2)
        # Reorder the dimensions to (variable, date, values)
        data = np.swapaxes(data, 0, 1)

        # apply reduction to `grid_indices`
        data = data[..., self.grid_indices]

        fields = {v: data[i] for i, v in enumerate(self.variables)}

        return dict(
            fields=fields,
            dates=dates,
            latitudes=self.latitudes,
            longitudes=self.longitudes,
        )

    def _load_dates(self, dates: List[Date]) -> Any:
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

    def __init__(
        self,
        context: Context,
        *args: Any,
        use_original_paths: bool = True,
        variables: Optional[List[str]],
        pre_processors: Optional[List[ProcessorConfig]] = None,
        grid_indices=None,
        **kwargs: Any,
    ) -> None:
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
        super().__init__(
            context,
            variables=variables,
            pre_processors=pre_processors,
            grid_indices=grid_indices,
            open_dataset_args=args,
            open_dataset_kwargs=kwargs,
        )


class DataloaderInput(DatasetInput):
    """Handles `anemoi-datasets` dataset as input."""

    def __init__(
        self,
        context: Context,
        *,
        use_original_paths: bool = True,
        **kwargs: Any,
    ) -> None:
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
        open_dataset_args, open_dataset_kwargs = context.checkpoint.open_dataset_args_kwargs(
            use_original_paths=use_original_paths,
            from_dataloader=self.name,
        )

        super().__init__(
            context,
            open_dataset_args=open_dataset_args,
            open_dataset_kwargs=open_dataset_kwargs,
            **kwargs,
        )


@input_registry.register("test")
class TestInput(DataloaderInput):
    """Handles `anemoi-datasets` dataset as input."""

    trace_name = "dataset/test"
    name = "test"


@input_registry.register("validation")
class ValidationInput(DataloaderInput):
    """Handles `anemoi-datasets` dataset as input."""

    trace_name = "dataset/validation"
    name = "validation"


@input_registry.register("training")
class TrainingInput(DataloaderInput):
    """Handles `anemoi-datasets` dataset as input."""

    trace_name = "dataset/training"
    name = "training"
