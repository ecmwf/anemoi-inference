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
from functools import cached_property
from typing import Any
from typing import Generator

from anemoi.utils.dates import frequency_to_timedelta as to_timedelta

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.lazy import torch
from anemoi.inference.runner import Runner
from anemoi.inference.runner import RunnerClasses
from anemoi.inference.tensors import Kind
from anemoi.inference.tensors import TensorHandler
from anemoi.inference.types import BoolArray

from . import runner_registry

LOG = logging.getLogger(__name__)


class FlexibleTensorHandler(TensorHandler):
    """TensorHandler for FlexibleForecaster checkpoints.

    Overrides the prognostic copy step to use the slot mapping computed by
    ``FlexibleRunner`` instead of the uniform roll-and-fill logic.
    """

    def copy_prognostic_fields_to_input_tensor(
        self,
        input_tensor_torch: torch.Tensor,
        y_pred: torch.Tensor,
        check: BoolArray,
    ) -> torch.Tensor:
        """Copy prognostic fields using the flexible slot mapping."""
        # input_tensor_torch shape: (batch, multi_step_input, values, variables)
        preserve_pairs, predict_pairs = self.context.slot_mapping

        pmask_in = torch.as_tensor(
            self.metadata.prognostic_input_mask,
            device=input_tensor_torch.device,
            dtype=torch.long,
        )
        pmask_out = torch.as_tensor(
            self.metadata.prognostic_output_mask,
            device=y_pred.device,
            dtype=torch.long,
        )

        prognostic_fields = torch.index_select(y_pred, dim=-1, index=pmask_out)

        new_tensor = input_tensor_torch.clone()
        for new_slot, old_slot in preserve_pairs:
            new_tensor[:, new_slot, :, :] = input_tensor_torch[:, old_slot, :, :]
        for new_slot, out_slot in predict_pairs:
            new_tensor[:, new_slot, :, pmask_in] = prognostic_fields[:, out_slot, :, :]

        pmask_in_np = pmask_in.detach().cpu().numpy()
        if check[pmask_in_np].any():
            conflicting = [self._input_tensor_by_name[i] for i in pmask_in_np[check[pmask_in_np]]]
            raise AssertionError(
                f"[{self.dataset_name}] Attempting to overwrite existing prognostic input slots "
                f"for variables: {conflicting}"
            )

        check[pmask_in_np] = True

        for n in pmask_in_np:
            self._input_kinds[self._input_tensor_by_name[n]] = Kind(prognostic=True)
            if self.trace:
                self.trace.from_rollout(self._input_tensor_by_name[n])

        return new_tensor


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

        super().__init__(config, classes=RunnerClasses(tensor_handler=FlexibleTensorHandler))

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

    @cached_property
    def slot_mapping(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Preserve and predict slot pairs for one AR step advance.

        After advancing by ``step_shift``, each new input slot at offset ``o``
        is filled from the old input slot at ``o + step_shift`` (preserve) or
        from the output slot at ``o + step_shift`` (predict).
        A tuple of ``(preserve_pairs, predict_pairs)`` where each element is a
        list of ``(new_input_slot, source_slot)`` index pairs.
        """
        input_index = {o: i for i, o in enumerate(self.input_offsets)}
        output_index = {o: i for i, o in enumerate(self.output_offsets)}
        preserve: list[tuple[int, int]] = []
        predict: list[tuple[int, int]] = []
        for new_slot, offset in enumerate(self.input_offsets):
            source = offset + self.step_shift
            if source in input_index:
                preserve.append((new_slot, input_index[source]))
            elif source in output_index:
                predict.append((new_slot, output_index[source]))
            else:
                raise ValueError(
                    f"Input offset {offset} + step_shift {self.step_shift} = {source} "
                    f"is neither in input_offsets {self.input_offsets} "
                    f"nor output_offsets {self.output_offsets}."
                )
        LOG.info(
            "FlexibleRunner slot_mapping: preserve=%s, predict=%s",
            preserve,
            predict,
        )
        return preserve, predict
