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
from collections.abc import Generator
from functools import cached_property
from typing import Any

import numpy as np
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from anemoi.utils.timer import Timer

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.inputs import create_input
from anemoi.inference.lazy import torch
from anemoi.inference.metadata import Metadata
from anemoi.inference.output import Output
from anemoi.inference.outputs import create_output
from anemoi.inference.profiler import ProfilingLabel
from anemoi.inference.runner import Runner
from anemoi.inference.types import FloatArray
from anemoi.inference.types import State

from . import runner_registry

LOG = logging.getLogger(__name__)


@runner_registry.register("spatial_downscaler")
class SpatialDownscalerRunner(Runner):
    """A runner for a residual spatial downscaler with multiple input and output steps.

    A spatial downscaler maps `k` input states on a coarse grid onto `k` output states
    on a fine grid, at the *same* valid times. It is therefore not autoregressive: the
    outputs are never fed back into the inputs, and each window of `k` steps is read
    afresh from the input source.

    The checkpoint is expected to hold one or more input datasets (the coarse fields
    and any fine-grid forcings) and one or more output datasets. Which is which comes
    from the role each dataset is recorded with in the metadata; the output datasets
    are never retrieved and are not passed to the model at all.

    Parameters
    ----------
    config : RunConfiguration
        The run configuration.
    window : str | None, optional
        How far to advance between two successive downscaling windows, e.g. ``"12h"``.
        Training does not determine this — it draws overlapping windows at the dataset
        frequency — so it is a choice made here. With several snapshots per window it
        defaults to the span the model was trained on, ``k * spacing``; with a single
        snapshot there is nothing to derive from and it has to be given, e.g.
        ``runner: {spatial_downscaler: {window: 6h}}``.
    """

    def __init__(self, config: RunConfiguration, *, window: str | None = None) -> None:
        super().__init__(config)

        if not self.checkpoint.multi_dataset:
            raise ValueError("The spatial downscaler runner requires a multi-dataset checkpoint.")

        self._validate_roles()

        # The window is how far the runner advances between model calls. The
        # snapshot spacing is the gap between the valid times inside one window.
        self.window = self._resolve_window(window)
        self.time_step = self.snapshot_spacing or self.window
        self.lead_time = to_timedelta(self.config.lead_time)

        LOG.info(
            "Spatial downscaler: inputs=%s, outputs=%s, window=%s (%s snapshot(s) at %s)",
            self.input_dataset_names,
            self.output_dataset_names,
            self.window,
            len(self.output_offsets),
            [str(offset) for offset in self.output_offsets],
        )

    #########################################################################################################
    # Dataset roles
    #########################################################################################################

    def _validate_roles(self) -> None:
        """Check the roles recorded in the checkpoint describe a downscaler."""
        if not self.output_dataset_names or self.output_dataset_names == self.dataset_names:
            raise ValueError(
                "The spatial downscaler runner needs the checkpoint to mark at least one dataset as an "
                "output and at least one as an input. Roles found: "
                f"{ {name: self.checkpoint.multi_dataset_metadata[name].role for name in self.dataset_names} }. "
                "Checkpoints from before training recorded roles can be fixed with `patch_metadata`, "
                "e.g. `metadata_inference: {out_hres: {role: output}}`."
            )

        if not self.input_dataset_names:
            raise ValueError("The spatial downscaler runner needs at least one input dataset.")

        multi_metadata = self.checkpoint.multi_dataset_metadata

        for name in self.output_dataset_names:
            if not multi_metadata[name].output_tensor_index_to_variable:
                raise ValueError(f"[{name}] Declared as an output dataset but the metadata has no output variables.")

        for name in self.input_dataset_names:
            if not multi_metadata[name].variable_to_input_tensor_index:
                raise ValueError(f"[{name}] Declared as an input dataset but the metadata has no input variables.")

        offsets = {name: multi_metadata[name].output_offsets for name in self.output_dataset_names}
        if len(set(tuple(o) for o in offsets.values())) > 1:
            raise ValueError(f"Output datasets have differing output offsets: {offsets}.")

    #########################################################################################################
    # Timing
    #########################################################################################################

    @cached_property
    def output_offsets(self) -> list[datetime.timedelta]:
        """Valid times produced by a single model call, relative to the start of the window."""
        return self.checkpoint.multi_dataset_metadata[self.output_dataset_names[0]].output_offsets

    @cached_property
    def snapshot_spacing(self) -> datetime.timedelta | None:
        """Gap between the valid times of one window, or ``None`` for a single snapshot."""
        offsets = self.output_offsets
        if len(offsets) < 2:
            return None

        spacings = {later - earlier for earlier, later in zip(offsets, offsets[1:])}
        if len(spacings) > 1:
            raise ValueError(
                f"The output offsets {[str(offset) for offset in offsets]} are not evenly spaced, so the "
                "downscaling window cannot be derived. Set `runner: {spatial_downscaler: {window: ...}}`."
            )

        return spacings.pop()

    def _resolve_window(self, configured: str | None) -> datetime.timedelta:
        """Decide how far to advance between successive windows."""
        spacing = self.snapshot_spacing

        if configured is None:
            if spacing is None:
                raise ValueError(
                    "The spatial downscaler runner cannot derive how far to advance between windows: the "
                    "model was trained on a single snapshot, and the checkpoint does not — and cannot — "
                    "record how often to apply it. Set `runner: {spatial_downscaler: {window: 6h}}`."
                )
            # Tile the sequence with the span the model was trained on.
            return len(self.output_offsets) * spacing

        window = to_timedelta(configured)
        if window <= datetime.timedelta(0):
            raise ValueError(f"The downscaling window must be positive, got {configured!r}.")

        if spacing is not None:
            span = self.output_offsets[-1] - self.output_offsets[0]
            if window < span + spacing:
                raise ValueError(
                    f"A window of {window} is shorter than the {span + spacing} covered by one model call "
                    f"({len(self.output_offsets)} snapshots at {spacing}), so successive windows would write "
                    "the same valid time twice."
                )
            if window % spacing:
                LOG.warning(
                    "The downscaling window %s is not a multiple of the %s snapshot spacing, so the output "
                    "times will not be evenly spaced.",
                    window,
                    spacing,
                )

        return window

    #########################################################################################################
    # I/O: nothing is retrieved for, or written from, the wrong side of the model
    #########################################################################################################

    def create_input(self, input_type: str, dataset_name: str, metadata: Metadata) -> Any:
        if dataset_name in self.output_dataset_names:
            return create_input(self, "empty", metadata, variables=[], purpose=input_type)

        return super().create_input(input_type, dataset_name, metadata)

    def create_output(self, dataset_name: str, metadata: Metadata) -> Output:
        if dataset_name not in self.output_dataset_names:
            return create_output(self, "none", metadata)

        return super().create_output(dataset_name, metadata)

    #########################################################################################################
    # Inference
    #########################################################################################################

    def forecast(
        self,
        lead_time: datetime.timedelta,
        input_tensors_numpy: dict[str, FloatArray],
        input_states: dict[str, State],
    ) -> Generator[dict[str, State], None, None]:
        """Downscale a single window of `k` input steps onto `k` output steps.

        Parameters
        ----------
        lead_time : datetime.timedelta
            Unused. This method processes exactly one window; `execute` drives the window loop.
        input_tensors_numpy : dict[str, FloatArray]
            The input tensors for each input dataset, with shape (multi_step_input, variables, values).
        input_states : dict[str, State]
            The input states for each input dataset.

        Yields
        ------
        dict[str, State]
            The downscaled states for each output dataset, one per output offset.
        """
        with torch.inference_mode():
            self.model.eval()

            input_tensors_torch = {
                dataset: torch.from_numpy(np.swapaxes(tensor, -2, -1)[np.newaxis, ...]).to(self.device)
                for dataset, tensor in input_tensors_numpy.items()
            }

            start = input_states[self.input_dataset_names[0]]["date"]

            for dataset in self.input_dataset_names:
                handler = self.tensor_handlers[dataset]
                if self.verbosity > 0:
                    handler._print_input_tensor(f"[{dataset}] Input tensor", input_tensors_torch[dataset])
                if handler.trace:
                    handler.trace.write_input_tensor(
                        start,
                        0,
                        input_tensors_torch[dataset].cpu().numpy(),
                        handler.metadata.variable_to_input_tensor_index,
                        self.time_step,
                    )

            amp_ctx = torch.autocast(device_type=self.device.type, dtype=self.autocast)

            with (
                torch.inference_mode(),
                amp_ctx,
                ProfilingLabel("Predict step", self.use_profiler),
                Timer(f"Downscaling {start} to {start + self.output_offsets[-1]}"),
            ):
                y_pred = self.predict_step(self.model, input_tensors_torch, fcstep=0, step=self.window, date=start)

            outputs = {dataset: self._squeeze_prediction(dataset, y_pred) for dataset in self.output_dataset_names}

            for i, offset in enumerate(self.output_offsets):
                states: dict[str, State] = {}

                for dataset in self.output_dataset_names:
                    handler = self.tensor_handlers[dataset]
                    output = outputs[dataset][i, ...]  # shape: (values, variables)

                    state: State = dict(
                        date=start + offset,
                        step=offset,
                        latitudes=handler.metadata.latitudes,
                        longitudes=handler.metadata.longitudes,
                        fields={},
                    )

                    for j in range(output.shape[1]):
                        state["fields"][handler.metadata.output_tensor_index_to_variable[j]] = output[:, j]

                    state, _ = self._apply_mid_processors(state, dataset)

                    if self.verbosity > 0:
                        handler._print_output_tensor(f"[{dataset}] Output tensor:", output.cpu().numpy())

                    if handler.trace:
                        handler.trace.write_output_tensor(
                            state["date"],
                            i,
                            output.cpu().numpy(),
                            handler.metadata.output_tensor_index_to_variable,
                            self.time_step,
                        )

                    states[dataset] = state

                yield states

    def _squeeze_prediction(self, dataset: str, y_pred: dict[str, "torch.Tensor"]) -> "torch.Tensor":
        """Reduce a model output to shape (time, values, variables)."""
        tensor = y_pred.get(dataset)

        if tensor is None:
            raise ValueError(
                f"[{dataset}] The model did not return a prediction for this dataset. "
                f"Datasets returned: {sorted(name for name, t in y_pred.items() if t is not None)}."
            )

        if tensor.ndim != 5:
            raise ValueError(
                f"[{dataset}] Output tensor should have dimensions (batch, time, ensemble, values, variables), "
                f"got {tuple(tensor.shape)}."
            )

        tensor = torch.squeeze(tensor, dim=(0, 2))

        if tensor.shape[0] != len(self.output_offsets):
            raise ValueError(
                f"[{dataset}] The model returned {tensor.shape[0]} output step(s) but the metadata declares "
                f"{len(self.output_offsets)} output offset(s): {self.output_offsets}."
            )

        return tensor

    #########################################################################################################
    def execute(self) -> None:
        """Downscale the requested lead time, one window of `k` steps at a time."""
        if self.config.description is not None:
            LOG.info("%s", self.config.description)

        num_windows = int(self.lead_time / self.window)

        if num_windows < 1:
            raise ValueError(f"Lead time {self.lead_time} is shorter than the downscaling window {self.window}.")

        if self.lead_time % self.window:
            LOG.warning(
                "Lead time %s is not a multiple of the downscaling window %s. Will downscale %s.",
                self.lead_time,
                self.window,
                num_windows * self.window,
            )

        # Constant forcings do not vary between windows, so they are only read once.
        constants_states: dict[str, State] = {}
        for dataset in self.input_dataset_names:
            constants_states[dataset] = self.constant_forcings_inputs[dataset].create_input_state(date=self.config.date)
            self._check_state(dataset, constants_states[dataset], "constant_forcings")

        self.input_states_hook(constants_states)

        opened: set[str] = set()

        for window_idx in range(num_windows):
            window_start = self.config.date + window_idx * self.window

            LOG.info("Downscaling window %s/%s starting at %s", window_idx + 1, num_windows, window_start)

            input_states: dict[str, State] = {}
            for dataset in self.input_dataset_names:
                prognostic_state = self.prognostics_inputs[dataset].create_input_state(date=window_start)
                self._check_state(dataset, prognostic_state, "prognostics")

                forcings_state = self.dynamic_forcings_inputs[dataset].create_input_state(date=window_start)
                self._check_state(dataset, forcings_state, "dynamic_forcings")

                constants_states[dataset]["date"] = window_start

                input_states[dataset] = self._combine_states(
                    prognostic_state,
                    constants_states[dataset],
                    forcings_state,
                )

            for states in self.run(input_states=input_states, lead_time=self.window):
                self.output_states_hook(states)

                for dataset, state in states.items():
                    state["step"] = state["step"] + window_idx * self.window

                    for processor in self.post_processors[dataset]:
                        state = processor.process(state)

                    if dataset not in opened:
                        self.outputs[dataset].open(state)
                        opened.add(dataset)

                    self.outputs[dataset].write_state(state)

        for output in self.outputs.values():
            output.close()
