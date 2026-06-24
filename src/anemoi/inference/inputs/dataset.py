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

from anemoi.inference.config.utils import multi_datasets_config
from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
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
        metadata: Metadata,
        *,
        open_dataset_args: tuple[Any, ...],
        open_dataset_kwargs: dict[str, Any],
        grid_indices: Any = None,
        use_trajectories: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the DatasetInput.

        Parameters
        ----------
        context : Any
            The context in which the input is used.
        metadata : Metadata
            Metadata corresponding to the dataset this input is handling.
        open_dataset_args : Tuple[Any, ...]
            Arguments for the dataset.
        open_dataset_kwargs : Dict[str, Any]
            Keyword arguments for the dataset.
        grid_indices : Optional[Any]
            Indices to reduce the input grid. If None, the full grid is used.
        use_trajectories : bool
            Whether to expect dataset as a trajectory (i.e. with a `step / forecast` dimension),
            and to get multiple dates from within a single trajectory.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(context, metadata, **kwargs)

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

        if grid_indices is None and "grid_indices" in self.metadata._supporting_arrays:
            grid_indices = self.metadata.load_supporting_array("grid_indices")
            if context.verbosity > 0:
                LOG.info("Loading supporting array `grid_indices` from checkpoint, \
                    the input grid will be reduced accordingly.")

        self.grid_indices = slice(None) if grid_indices is None else grid_indices

        self.use_trajectories = use_trajectories

        if not len(self.ds.shape) == 5 and self.use_trajectories:
            raise ValueError(
                f"Expected dataset with 5 dimensions (base_dates, variables, ensembles, steps, cells) as a trajectory dataset, got {len(self.ds.shape)} dimensions. Is this a trajectory dataset?"
            )

    @cached_property
    def ds(self) -> Any:
        """Return the dataset."""
        from anemoi.datasets import open_dataset

        LOG.info("Opening dataset...")
        LOG.info("open_dataset_args: %s", json.dumps(self.open_dataset_args, indent=2))
        LOG.info("open_dataset_kwargs: %s", json.dumps(self.open_dataset_kwargs, indent=2))

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

    def create_input_state(self, *, date: Date | None = None, constant: bool = False, **kwargs) -> State:
        """Create the input state for the given date.

        Parameters
        ----------
        date : Optional[Any]
            The date for which to create the input state.
        constant: bool
            Whether the field is constant or dynamic
        **kwargs : Any
            Additional keyword arguments.

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

        if constant:
            dates = [date]
        else:
            dates = [date + np.timedelta64(h) for h in self.metadata.lagged]

        data = self._load_dates(dates, base_date=date)

        if data.shape[2] != 1:
            raise ValueError(f"Ensemble data not supported, got {data.shape[2]} members")

        requested_variables = set(self.input_variables())
        dataset_variables = {}
        typed_variables = self.ds.typed_variables
        for i, variable in enumerate(self.ds.variables):
            if variable not in requested_variables:
                continue
            # Squeeze the data to remove the ensemble dimension
            values = np.squeeze(data[:, i], axis=1)
            fields[variable] = values[:, self.grid_indices]
            dataset_variables[variable] = typed_variables[variable]

            if trace := self.context.tensor_handlers[self.dataset_name].trace:
                trace.from_input(variable, self)

        input_state["_input"] = self
        input_state["_variables"] = dataset_variables

        return input_state

    def load_forcings_state(self, *, dates: list[Date], current_state: State) -> State:
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
        data = self._load_dates(dates, base_date=current_state["date"])  # (date, variables, ensemble, values)

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

    def _find_index_for_date(self, date: Date) -> int:
        """Find the index of the given date in the dataset.

        Parameters
        ----------
        date : Any
            The date for which to find the index.

        Returns
        -------
        int
            The index of the date in the dataset.
        """
        (i,) = np.where(self.ds.base_dates == np.datetime64(date))
        if len(i) == 0:
            raise ValueError(
                f"Date {date} not found in the dataset, available base_dates: {self.ds.base_dates[0]}...{self.ds.base_dates[-1]}"
            )
        assert len(i) == 1, f"Multiple dates found for {date}"
        return int(i[0])

    def _load_basedates(self, basedates: list[Date]) -> Any:
        """Load the data for the given base dates.

        Parameters
        ----------
        basedates : List[Any]
            List of base dates for which to load the data.

        Returns
        -------
        Any
            The loaded data.
        """
        # TODO: use the fact that the dates are sorted
        idx = []
        for d in basedates:
            idx.append(self._find_index_for_date(d))

        if len(idx) == 1:
            s = slice(idx[0], idx[0] + 1)
        else:
            diff = idx[1] - idx[0]
            if not all(i == diff for i in np.diff(idx)):
                # TODO: remove that restriction
                raise ValueError("Dates do not have the same frequency")
            s = slice(idx[0], idx[-1] + 1, diff)

        return self.ds[s]

    def _load_trajectories(self, dates: list[Date], base_date: Date) -> Any:
        """Load the data for the given dates as trajectories.

        Parameters
        ----------
        dates : List[Any]
            List of dates for which to load the data.
        base_date : Any
            The base date for relative date calculations.

        Returns
        -------
        Any
            The loaded data.
        """
        base_idx = self._find_index_for_date(base_date)
        step_index = [int(np.timedelta64(d - base_date) / self.ds.step_frequency) for d in dates]
        LOG.info("Loading trajectory data for base_date=%s, steps=%s", base_date, step_index)
        # Convert to slice if consecutive (dataset indexing with lists can be unreliable)
        if len(step_index) == 1:
            step_slice = slice(step_index[0], step_index[0] + 1)
        else:
            diff = step_index[1] - step_index[0]
            if all(step_index[i + 1] - step_index[i] == diff for i in range(len(step_index) - 1)):
                step_slice = slice(step_index[0], step_index[-1] + 1, diff)
            else:
                raise ValueError("Requested dates do not have a uniform step spacing")
        data = self.ds[base_idx, :, :, step_slice]
        # Transpose to (steps/dates, variables, ensemble, cells) to match _load_basedates
        data = np.moveaxis(data, 2, 0)
        return data

    def _load_dates(self, dates: list[Date], base_date: Date) -> Any:
        """Load the data for the given dates.

        Parameters
        ----------
        dates : List[Any]
            List of dates for which to load the data.
        base_date : Any
            The base date for relative date calculations.

        Returns
        -------
        Any
            The loaded data.
        """
        if not self.use_trajectories:
            return self._load_basedates(dates)
        return self._load_trajectories(dates, base_date=base_date)


@input_registry.register("dataset")
class DatasetInputArgsKwargs(DatasetInput):
    """Handles `anemoi-datasets` dataset as input."""

    trace_name = "dataset/provided"

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *args: Any,
        use_original_paths: bool = False,
        variables: list[str] | None,
        pre_processors: list[ProcessorConfig] | None = None,
        grid_indices=None,
        purpose: str | None = None,
        use_trajectories: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the DatasetInputArgsKwargs.

        Parameters
        ----------
        context : Context
            The context in which the input is used.
        metadata : Metadata
            Metadata corresponding to the dataset this input is handling.
        use_original_paths : bool
            Whether to use original paths.
        use_trajectories : bool
            Whether to expect dataset as a trajectory (i.e. with a `step / forecast` dimension),
            and to get multiple dates from within a single trajectory.
        """

        check_variables_compatibility = multi_datasets_config(
            context.config.check_variables_compatibility,
            metadata.dataset_name,
            context.dataset_names,
        )

        if not args and not kwargs:
            args, kwargs = metadata.open_dataset_args_kwargs(use_original_paths=use_original_paths)

            # TODO: remove start/end from the arguments

            LOG.warning("No arguments provided to open_dataset, using the default arguments:")

            cmd = "open_dataset("
            for arg in args:
                cmd += f"{arg}, "
            for key, value in kwargs.items():
                cmd += f"{key}={value}, "

            LOG.warning("%s", cmd)

        if check_variables_compatibility:
            kwargs = kwargs.copy()
            kwargs["check_variables_compatibility"] = check_variables_compatibility

        super().__init__(
            context,
            metadata,
            variables=variables,
            pre_processors=pre_processors,
            grid_indices=grid_indices,
            open_dataset_args=args,
            open_dataset_kwargs=kwargs,
            purpose=purpose,
            use_trajectories=use_trajectories,
        )


class DataloaderInput(DatasetInput):
    """Handles `anemoi-datasets` dataset as input."""

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        use_original_paths: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the DataloaderInput.

        Parameters
        ----------
        context : Any
            The context in which the input is used.
        metadata : Metadata
            The metadata corresponding to the dataset this input is handling.

        name : str
            The name of the dataloader.
        use_original_paths : bool
            Whether to use original paths.
        """
        open_dataset_args, open_dataset_kwargs = metadata.open_dataset_args_kwargs(
            use_original_paths=use_original_paths,
            from_dataloader=self.name,
        )

        super().__init__(
            context,
            metadata,
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
