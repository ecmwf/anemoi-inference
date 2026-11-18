# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging

import numpy as np
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta

from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.types import Date
from anemoi.inference.types import FloatArray
from anemoi.inference.types import State

from ..processor import Processor
from . import post_processor_registry

LOG = logging.getLogger(__name__)


@post_processor_registry.register("night_mask")
class NightMask(Processor):
    """Set accumulated radiation fields to zero where the sun was down for the whole accumulation window.

    Each field is an accumulation over the period ending at the state's date ``t``, so it can only be non-zero
    if the sun was up at some point in ``(t - accumulation_period, t]``. A point is therefore set to zero where
    the mask variable (by default ``insolation``) is ``<= 0`` both at ``t`` and at ``t - accumulation_period``.

    This is the inference counterpart of the ``night_mask`` option of the training ``TemporalDownscaler`` task
    and gives identical results: the mask variable is computed with the same code that provides the model's
    computed forcings.

    Only states produced by the temporal downscaler (``state["temporally_downscaled"]`` is true) are masked.
    The boundary states (t=0, and the inputs) are data, not model output, and are left untouched.

    This processor must come **before** any processor that accumulates the fields, such as
    ``accumulate_from_start_of_forecast``.

    Parameters
    ----------
    context : Context
        The context in which the processor is running.
    metadata : Metadata
        Metadata corresponding to the dataset this processor is handling.
    variables : list[str]
        The fields to set to zero at night.
    mask_variable : str, optional
        The computed forcing used as mask, by default "insolation".
    accumulation_period : str or int, optional
        The period over which the fields are accumulated, by default the data frequency of the checkpoint
        (the output timestep of a temporal downscaler).
    """

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        variables: list[str],
        mask_variable: str = "insolation",
        accumulation_period: str | int | None = None,
    ) -> None:
        super().__init__(context, metadata)

        if not variables:
            raise ValueError("night_mask: `variables` must list at least one field.")

        self.variables = list(variables)
        self.mask_variable = mask_variable
        self.accumulation_period = to_timedelta(
            accumulation_period if accumulation_period is not None else metadata.data_frequency
        )
        if self.accumulation_period <= datetime.timedelta(0):
            raise ValueError(f"night_mask: accumulation_period must be positive, got {self.accumulation_period}.")

        self._cache: dict[datetime.datetime, FloatArray] = {}
        self._cached_grid: tuple[int, float, float] | None = None
        LOG.info(
            "Night mask on %s where %s <= 0 at t and t-%s",
            self.variables,
            self.mask_variable,
            self.accumulation_period,
        )

    def _mask_field(self, state: State, date: Date) -> FloatArray:
        """Return the mask variable on the state's grid at ``date``, with a small cache of recent dates."""
        lat, lon = state["latitudes"], state["longitudes"]
        grid = (lat.size, float(lat[0]), float(lon[0]))
        if grid != self._cached_grid:  # new grid: drop everything
            self._cache.clear()
            self._cached_grid = grid

        if date not in self._cache:
            if len(self._cache) >= 4:  # outputs advance in time, only the last few dates are reused
                self._cache.pop(next(iter(self._cache)))
            self._cache[date] = self._compute(lat, lon, date)
        return self._cache[date]

    def _compute(self, lat: FloatArray, lon: FloatArray, date: Date) -> FloatArray:
        """Compute the mask variable the same way as ``ComputedForcings.load_forcings_array``."""
        import earthkit.data as ekd
        from anemoi.transform.grids.unstructured import UnstructuredGridFieldList

        source = UnstructuredGridFieldList.from_values(latitudes=lat, longitudes=lon)
        ds = ekd.from_source("forcings", source, date=[date], param=[self.mask_variable])
        assert len(ds) == 1, (len(ds), date)
        return ds[0].to_numpy(dtype=np.float32, flatten=True)

    def process(self, state: State) -> State:
        """Zero the configured fields where it was night during the whole accumulation window.

        Parameters
        ----------
        state : State
            The state containing the fields.

        Returns
        -------
        State
            The state with the fields masked.
        """
        if not state.get("temporally_downscaled", False):
            return state

        missing = [v for v in self.variables if v not in state["fields"]]
        if missing:
            raise ValueError(f"night_mask: fields {missing} are not in the state (has {sorted(state['fields'])}).")

        date = state["date"]
        night = (self._mask_field(state, date) <= 0) & (self._mask_field(state, date - self.accumulation_period) <= 0)

        state = state.copy()
        state["fields"] = state["fields"].copy()
        for name in self.variables:
            field = state["fields"][name]
            state["fields"][name] = np.where(night, np.zeros((), dtype=field.dtype), field)
        return state

    def __repr__(self) -> str:
        """Return a string representation of the NightMask object.

        Returns
        -------
        str
            A string representation of the NightMask object.
        """
        return (
            f"NightMask({self.variables}, mask_variable={self.mask_variable}, "
            f"accumulation_period={self.accumulation_period})"
        )
