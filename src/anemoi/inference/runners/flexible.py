# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import datetime
import logging
import math
from typing import Any
from typing import Generator

from anemoi.utils.dates import frequency_to_timedelta as to_timedelta

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runner import Runner
from anemoi.inference.runner import RunnerClasses

from . import runner_registry

LOG = logging.getLogger(__name__)


@runner_registry.register("flexible")
class FlexibleRunner(Runner):
    """Runner for FlexibleForecaster checkpoints.

    All temporal structure (input/output offsets, step shift) is supplied via
    the inference YAML config.  The runner does **not** rely on
    ``checkpoint.timestep`` (which is ``None`` for FlexibleForecaster
    checkpoints).
    """

    def __init__(
        self,
        config: RunConfiguration,
        input_offsets: list[str],
        output_offsets: list[str],
        step_shift: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the FlexibleRunner.

        Parameters
        ----------
        config : RunConfiguration
            Standard inference run configuration.
        input_offsets : list[str]
            Duration strings for input time slots, e.g. ``["-6H", "0H"]``.
        output_offsets : list[str]
            Duration strings for output time slots, e.g. ``["6H"]``.
        step_shift : str | None, optional
            Autoregressive step shift.  If ``None``, inferred as
            ``max(output_offsets) - max(input_offsets)``.
        **kwargs : Any
            Additional keyword arguments (ignored with a warning).
        """
        if kwargs:
            LOG.warning("FlexibleRunner: ignoring unknown runner options: %s", list(kwargs))

        super().__init__(config, classes=RunnerClasses())

        self.input_offsets = sorted(to_timedelta(x) for x in input_offsets)
        self.output_offsets = sorted(to_timedelta(x) for x in output_offsets)
        self.step_shift = (
            to_timedelta(step_shift) if step_shift is not None else max(self.output_offsets) - max(self.input_offsets)
        )

        # Override metadata.lagged on every dataset so that all input classes
        # and add_initial_forcings_to_input_state fetch the correct historical
        # dates instead of the uniform [-i*timestep] sequence.
        for metadata in self.checkpoint.multi_dataset_metadata.values():
            metadata.lagged = self.input_offsets

        LOG.info(
            "FlexibleRunner: input_offsets=%s, output_offsets=%s, step_shift=%s",
            self.input_offsets,
            self.output_offsets,
            self.step_shift,
        )

    def create_output(self, dataset_name: str, metadata: Any) -> Any:
        """Create an output object, raising if zarr is requested."""
        from anemoi.inference.outputs.zarr import ZarrOutput

        output = super().create_output(dataset_name, metadata)
        if isinstance(output, ZarrOutput):
            raise NotImplementedError(
                "FlexibleRunner does not support zarr output yet [time_step=None]. "
                "Use a different output type (e.g. grib)."
            )
        return output

    def forecast_stepper(
        self, start_date: datetime.datetime, lead_time: datetime.timedelta
    ) -> Generator[tuple[datetime.timedelta, list[datetime.datetime], list[datetime.datetime], bool], None, None]:
        """Generate step and date variables for the flexible forecast autoregressive loop."""
        steps = math.ceil(lead_time / self.step_shift)

        LOG.info(
            "FlexibleRunner: lead_time=%s, step_shift=%s, output_offsets=%s, forecasting %s steps.",
            lead_time,
            self.step_shift,
            self.output_offsets,
            steps,
        )

        for s in range(steps):
            step = (s + 1) * self.step_shift
            valid_dates = [start_date + s * self.step_shift + offset for offset in self.output_offsets]
            next_dates = valid_dates
            is_last_step = s == steps - 1
            yield step, valid_dates, next_dates, is_last_step
