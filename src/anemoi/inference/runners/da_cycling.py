# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""DA Cycling Runner for inference (multi-dataset aware).

This runner extends the default Runner with Data Assimilation cycling before
the forecast rollout. The DA cycles happen *before* the user's requested date
so that the analysis state lands on that date and the forecast starts there.

For example, with ``date: 2020-06-01``, ``da_cycles: 2``, ``timestep: 6h``:
- Input loaded at 2020-05-31 12:00 (date - 2*6h)
- DA cycle 1: predict -> blend obs at 2020-05-31 18:00
- DA cycle 2: predict -> blend obs at 2020-06-01 00:00
- Analysis = 2020-06-01 00:00 (the user's requested date)
- Forecast starts from 2020-06-01 00:00

Multi-step-output checkpoints (``multistep_output > 1``) are supported: each DA
cycle is one model call advancing ``multi_step_output`` timesteps, and every
predicted step is blended with observations valid at its own date. The input
date is shifted back by ``da_cycles * multi_step_output * timestep`` accordingly.

Configuration example for a multi-dataset checkpoint::

    runner: da_cycling
    da_cycling:
      da_cycles: 2
      observation_sources:
        conventional: { ... input config ... }
        polar_sat:    { ... input config ... }

For a single-dataset checkpoint ``observation_source`` (singular) is also
accepted and is mapped to the sole dataset name. ``da_cycles`` is optional and
defaults to the number of cycles the model was trained with, read from the
checkpoint metadata (``config.task.da_cycles``); an explicit 0 disables cycling
and the runner behaves exactly like the default runner.

Correspondence with training (``anemoi.training.tasks.da_forecaster.DAForecaster``
driven by ``DASingleTraining``):

- DA cycle: the new input frame takes observation values where they exist and
  falls back to the raw model prediction for prognostic variables where the
  observation is NaN. Corrector variables (satellite viewing geometry,
  reportype, ...) keep their observed values during DA cycles.
- Forecast step: prognostics come from the prediction, forcings from the forcing
  providers, and corrector slots are reset so the checkpoint's imputer applies
  the same no-obs sentinel the model was trained with.
- The per-instrument corrector *network* is deliberately not reproduced here. It
  exists only to shape the training loss, lives on the Lightning module rather
  than the model interface, and is therefore absent from inference checkpoints.
  Only the corrector *variables* matter at inference, and they are supplied by
  the observation source.
- The input window stays at exactly ``multi_step_input`` frames, which keeps the
  checkpoint's ``RandomSpatialDropout`` disabled (it self-disables when the time
  dimension does not exceed its ``multi_step``) and the imputer in its
  inference-time behaviour.
"""

from __future__ import annotations

import datetime
import logging
from collections.abc import Generator
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from anemoi.inference.config.utils import multi_datasets_config
from anemoi.inference.input import Input
from anemoi.inference.inputs import create_input
from anemoi.inference.lazy import torch
from anemoi.inference.runners.default import DefaultRunner
from anemoi.inference.types import FloatArray
from anemoi.inference.types import State

from . import runner_registry

if TYPE_CHECKING:
    from anemoi.inference.config.run import RunConfiguration

LOG = logging.getLogger(__name__)


@runner_registry.register("da_cycling")
class DACyclingRunner(DefaultRunner):
    """Runner with DA cycling before forecast.

    Performs ``da_cycles`` assimilation steps at the start of inference,
    blending model predictions with loaded observations to produce an analysis
    state. The forecast then rolls out from this analysis.
    """

    def __init__(self, config: "RunConfiguration") -> None:
        super().__init__(config)

        da_config = getattr(config, "da_cycling", None) or {}
        if not isinstance(da_config, dict):
            # Pydantic model -> dict
            da_config = dict(da_config)

        # Default to the number of DA cycles the model was trained with. The
        # checkpoint's `timesteps` metadata describes a single input/output window and
        # no longer spans the DA + rollout window, so the training task configuration
        # is the only surviving source. An explicit 0 in the config disables DA cycling.
        configured_cycles = da_config.get("da_cycles")
        if configured_cycles is None:
            self.da_cycles: int = int(self.checkpoint.da_cycles or 0)
            cycles_source = "checkpoint metadata (config.task.da_cycles)"
        else:
            self.da_cycles = int(configured_cycles or 0)
            cycles_source = "run configuration"

        self._obs_source_configs: dict[str, Any] = self._resolve_observation_sources(da_config)

        if self.da_cycles > 0:
            LOG.info(
                "DA Cycling Runner: %d DA cycles before forecast, from %s (datasets: %s)",
                self.da_cycles,
                cycles_source,
                list(self._obs_source_configs.keys()),
            )
        else:
            LOG.info("DA Cycling Runner: da_cycles=0 from %s, running as a plain forecast", cycles_source)

        self._obs_inputs: dict[str, Input] | None = None

    # ── Observation source configuration ──────────────────────────────

    def _resolve_observation_sources(self, da_config: dict) -> dict[str, Any]:
        """Resolve the per-dataset observation source configs.

        Accepts both spellings (``observation_source`` / ``observation_sources``).
        The value follows the same convention as ``input`` / ``output`` in the rest
        of the runner: a config keyed by dataset names is treated as per-dataset,
        any other config is broadcast to every dataset.
        """
        sources = da_config.get("observation_sources")
        single = da_config.get("observation_source")

        if sources is not None and single is not None:
            raise ValueError(
                "da_cycling: specify either `observation_sources` or "
                "`observation_source`, not both.",
            )

        obs_config = sources if sources is not None else single
        if obs_config is None:
            return {}

        return {
            ds: multi_datasets_config(obs_config, ds, self.dataset_names, strict=True)
            for ds in self.dataset_names
        }

    # ── Observation input creation / loading ──────────────────────────

    def _create_observation_inputs(self) -> dict[str, Input]:
        """Create one observation Input per dataset that has a source configured."""
        if not self._obs_source_configs:
            raise ValueError(
                "da_cycling.observation_sources must be configured when da_cycles > 0",
            )

        multi_metadata = self.checkpoint.multi_dataset_metadata
        inputs: dict[str, Input] = {}
        for ds, cfg in self._obs_source_configs.items():
            metadata = multi_metadata[ds]
            # Variables to assimilate: the dataset's prognostic input variables,
            # plus the corrector variables (satellite viewing geometry, reportype,
            # ...) that describe the observations. Training keeps real corrector
            # values in the blended DA state (DAGraphForecaster._da_blend starts
            # from obs.clone()), so they must be loaded with the observations.
            variables = [
                metadata.input_tensor_index_to_variable[i]
                for i in metadata.prognostic_input_mask
            ]
            variables += [v for v in metadata.corrector_variables if v not in variables]
            inputs[ds] = create_input(
                self,
                cfg,
                metadata,
                variables=variables,
                purpose="da_observations",
            )
            LOG.info("[%s] DA observation input: %s (variables=%d)", ds, inputs[ds], len(variables))
        return inputs

    def _load_observations(
        self, date: datetime.datetime, states: dict[str, State]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Load observations for a given DA target date, per dataset.

        Returns a mapping ``dataset_name -> {variable_name: array}``. Datasets
        without an observation source produce an empty dict.
        """
        if self._obs_inputs is None:
            self._obs_inputs = self._create_observation_inputs()

        result: dict[str, dict[str, np.ndarray]] = {}
        for ds in self.dataset_names:
            if ds not in self._obs_inputs:
                result[ds] = {}
                continue
            obs_state = self._obs_inputs[ds].create_input_state(date=date)
            result[ds] = obs_state.get("fields", {})
        return result

    # ── DA blend ───────────────────────────────────────────────────────

    @staticmethod
    def _prognostic_fields(y_pred: "torch.Tensor", metadata: Any) -> "torch.Tensor":
        """Extract the prognostic columns of a prediction as ``(batch, time, grid, vars)``.

        Parameters
        ----------
        y_pred : torch.Tensor
            Model output for one dataset, shape
            ``(batch, [time], [ensemble], grid, n_out_vars)``.
        metadata : Any
            The dataset's tensor-handler metadata, read for
            ``prognostic_output_mask``.

        Returns
        -------
        torch.Tensor
            Prognostic fields, shape ``(batch, time, grid, n_prognostic)``.
        """
        pmask_out = torch.as_tensor(
            metadata.prognostic_output_mask,
            device=y_pred.device,
            dtype=torch.long,
        )
        prognostic_fields = torch.index_select(y_pred, dim=-1, index=pmask_out)

        # Normalize to (batch, time, grid, vars)
        if prognostic_fields.ndim == 4:  # pre-multistep models: (batch, ensemble, grid, vars)
            prognostic_fields = prognostic_fields.unsqueeze(1)
        # Drop any remaining ensemble dim: (batch, time, ensemble, grid, vars)
        while prognostic_fields.ndim > 4:
            prognostic_fields = prognostic_fields.squeeze(2)

        return prognostic_fields

    def _da_blend(
        self,
        input_tensors_torch: dict[str, "torch.Tensor"],
        y_preds: dict[str, "torch.Tensor"],
        obs_per_dataset: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, "torch.Tensor"]:
        """Blend model prediction with observations into the input tensors per dataset.

        Mirrors the training-time DA blend (DAGraphForecaster._da_blend), including
        multi-step output: the window is rolled forward by
        ``keep_steps = min(multi_step_output, multi_step_input)`` and each of the
        last ``keep_steps`` slots receives its own blended state — the prediction
        step valid at that slot's date, blended with observations valid at that
        same date. Per slot:

        - Prognostic variables: use observation where available (not NaN),
          use model prediction where observation is NaN or missing.
        - Corrector slots are reset to NaN then filled with the slot's observed
          geometry where present.
        - Other observation fields (forcings, ...) are written where observed.

        Parameters
        ----------
        input_tensors_torch : dict[str, torch.Tensor]
            Per-dataset input tensors, each shape (1, multi_step_input, grid, variables).
        y_preds : dict[str, torch.Tensor]
            Per-dataset model predictions, each shape
            (1, multi_step_output, ensemble, grid, n_out_vars).
        obs_per_dataset : dict[str, dict[str, np.ndarray]]
            Per-dataset observation fields keyed by variable name. Multi-row
            fields must be spaced by the checkpoint timestep and end at the
            cycle end date (as returned by ``create_input_state`` at that date).

        Returns
        -------
        dict[str, torch.Tensor]
            Updated per-dataset input tensors with blended analysis states in
            the last ``keep_steps`` slots.
        """
        out: dict[str, "torch.Tensor"] = {}
        for ds, input_tensor_torch in input_tensors_torch.items():
            metadata = self.tensor_handlers[ds].metadata
            y_pred = y_preds[ds]

            pmask_in = torch.as_tensor(
                metadata.prognostic_input_mask,
                device=input_tensor_torch.device,
                dtype=torch.long,
            )
            # Called on the class, not self: _da_blend is driven unbound against a
            # lightweight stub by the training/inference parity harness.
            prognostic_fields = DACyclingRunner._prognostic_fields(y_pred, metadata)

            multi_step_input = input_tensor_torch.shape[1]
            n_time = prognostic_fields.shape[1]
            keep_steps = min(n_time, multi_step_input)

            # Roll the input window forward by the number of steps we keep
            input_tensor_torch = input_tensor_torch.roll(-keep_steps, dims=1)

            # Corrector slots (satellite viewing geometry, reportype, ...) are
            # reset to NaN before writing each slot's observations: geometry from
            # a previous timestep must not survive the roll where the new cycle
            # has no observation. NaN becomes the no-obs sentinel via the
            # checkpoint's imputer, matching training where the blended state
            # carries the (imputed) obs corrector values at the DA target time.
            cmask_in = metadata.corrector_input_mask
            cmask = (
                torch.as_tensor(cmask_in, device=input_tensor_torch.device, dtype=torch.long)
                if len(cmask_in) > 0
                else None
            )

            # In training the blend starts from obs.clone() which carries ALL observed
            # variables at the target time; here we mimic that by writing every observed
            # field that maps to an input-tensor slot.
            variable_to_input = metadata.variable_to_input_tensor_index
            obs_tensors: dict[str, "torch.Tensor"] = {}
            for var_name, obs_array in obs_per_dataset.get(ds, {}).items():
                if var_name not in variable_to_input:
                    continue
                obs_tensors[var_name] = torch.from_numpy(obs_array).to(
                    device=input_tensor_torch.device,
                    dtype=input_tensor_torch.dtype,
                )

            for i in range(keep_steps):
                slot = -(keep_steps - i)  # input-tensor slot to fill
                t_pred = n_time - keep_steps + i  # model output step valid at this slot's date

                # Background fill from the prediction
                input_tensor_torch[:, slot, :, pmask_in] = prognostic_fields[:, t_pred]

                if cmask is not None:
                    input_tensor_torch[:, slot, :, cmask] = torch.nan

                # Overwrite with observations where available (incl. satellite geometry)
                for var_name, obs_tensor in obs_tensors.items():
                    idx = variable_to_input[var_name]
                    # Multi-row obs end at the cycle end date and are spaced by
                    # timestep, so the row valid at this slot's date is the same
                    # negative index as the slot. Single-row obs are used as-is
                    # (only exact for keep_steps == 1).
                    obs_row = obs_tensor[slot] if obs_tensor.ndim > 1 else obs_tensor

                    mask = ~torch.isnan(obs_row)
                    if mask.any():
                        input_tensor_torch[:, slot, mask, idx] = obs_row[mask]

            out[ds] = input_tensor_torch
        return out

    # ── Date shifting ────────────────────────────────────────────────

    def execute(self) -> None:
        """Execute with date shifted back so DA cycles land on the user's date.

        Input is loaded at ``date - da_cycles * multi_step_output * timestep``
        (each DA cycle is one model call and advances ``multi_step_output``
        timesteps). DA cycles then advance forward to reach the original date,
        and the forecast rolls out from there.
        """
        if self.da_cycles == 0:
            super().execute()
            return

        timestep = self.checkpoint.timestep
        n_out = self.checkpoint.multi_step_output
        original_date = self.config.date
        shifted_date = original_date - self.da_cycles * n_out * timestep

        LOG.info(
            "DA cycling: shifting input date from %s back to %s (%d cycles x %d step(s) x %s)",
            original_date,
            shifted_date,
            self.da_cycles,
            n_out,
            timestep,
        )

        # Shift date back for input loading
        self.config.date = shifted_date

        # Suppress initial-state write — the pre-DA input is not the true analysis;
        # the forecast output starts from the analysis date.
        #
        # Both the config field and the instance attribute have to be overridden:
        # Runner.__init__ has already copied the config value onto self, and
        # Output.write_step_zero reads the *instance* attribute via the context,
        # so setting the config alone would have no effect.
        original_write_initial = self.config.write_initial_state
        original_write_initial_attr = self.write_initial_state
        self.config.write_initial_state = False
        self.write_initial_state = False

        # Note: self.reference_date was captured at __init__ from the original
        # (unshifted) date. That is the analysis date, which is exactly the
        # forecast_reference_time the outputs should carry, so it is left alone.

        try:
            super().execute()
        finally:
            self.config.date = original_date
            self.config.write_initial_state = original_write_initial
            self.write_initial_state = original_write_initial_attr

    # ── Forecast with DA cycling ──────────────────────────────────────

    def forecast(
        self,
        lead_time: str,
        input_tensors_numpy: dict[str, FloatArray],
        input_states: dict[str, State],
    ) -> Generator[dict[str, State], None, None]:
        """Forecast with DA cycling before the rollout.

        Parameters
        ----------
        lead_time : str
            The lead time for the forecast.
        input_tensors_numpy : dict[str, FloatArray]
            Per-dataset input tensors, each with shape (multi_step_input, variables, values).
        input_states : dict[str, State]
            Per-dataset input states.
        """
        if self.da_cycles == 0:
            yield from super().forecast(lead_time, input_tensors_numpy, input_states)
            return

        with torch.inference_mode():
            self.model.eval()

            # Convert each dataset's input tensor to torch shape (1, multi_step_input, grid, variables)
            input_tensors_torch: dict[str, "torch.Tensor"] = {
                ds: torch.from_numpy(np.swapaxes(arr, -2, -1)[np.newaxis, ...]).to(self.device)
                for ds, arr in input_tensors_numpy.items()
            }

            start_date = next(iter(input_states.values()))["date"]
            timestep = self.checkpoint.timestep

            LOG.info(
                "Running %d DA cycles from %s (timestep=%s, datasets=%s)",
                self.da_cycles,
                start_date,
                timestep,
                self.dataset_names,
            )

            # Pre-compute per-dataset typed-variable masks for the
            # `check` array consumed by add_dynamic_forcings_to_input_tensor.
            # Same rule as the base forecast loop: constants never need refreshing,
            # and diagnostic input columns (which is where correctors land, since
            # they carry no standard category) are written by the DA blend rather
            # than by a forcing provider.
            constant_check_template: dict[str, np.ndarray] = {}
            for ds, handler in self.tensor_handlers.items():
                mask = np.full((input_tensors_torch[ds].shape[-1],), False)
                typed_variables = handler.metadata.typed_variables
                categories = handler.metadata.variable_categories()
                for variable, i in handler.metadata.variable_to_input_tensor_index.items():
                    if typed_variables[variable].is_constant_in_time:
                        mask[i] = True
                    elif "diagnostic" in categories.get(variable, []):
                        mask[i] = True
                constant_check_template[ds] = mask

            # ── DA cycling phase ──────────────────────────────────────
            # Each cycle is one model call and advances multi_step_output timesteps.
            n_out = self.checkpoint.multi_step_output
            for cycle in range(self.da_cycles):
                # Valid dates of this cycle's predicted output steps
                da_target_dates = [start_date + (cycle * n_out + t + 1) * timestep for t in range(n_out)]
                cycle_end_date = da_target_dates[-1]
                LOG.info(
                    "DA cycle %d/%d: target date(s) %s",
                    cycle + 1,
                    self.da_cycles,
                    ", ".join(str(d) for d in da_target_dates),
                )

                # Build per-dataset states at the cycle end date (used by forcings providers).
                #
                # `step` is the elapsed lead time since the run's own start, the same
                # convention the base forecast loop uses, so that `date - step` recovers
                # the base date the input was loaded at. Dataset inputs rely on that to
                # resolve relative dates (e.g. when opened with use_trajectories).
                da_states: dict[str, State] = {}
                for ds in self.dataset_names:
                    s = input_states[ds].copy()
                    s["date"] = cycle_end_date
                    s["previous_step"] = s.get("step")
                    s["step"] = cycle_end_date - start_date
                    da_states[ds] = s

                # Load decoder-forcings for the predicted dates (per dataset)
                decoder_forcings: dict[str, "torch.Tensor"] = {}
                for ds, handler in self.tensor_handlers.items():
                    df_tensor = handler.load_decoder_forcings(da_states[ds], da_target_dates)
                    if df_tensor is not None:
                        decoder_forcings[ds] = df_tensor

                # fcstep/step/date/decoder_forcings are set by this runner; a user
                # `predict_kwargs` entry colliding with them is warned about by
                # Runner.predict_step.
                predict_kwargs: dict[str, Any] = dict(
                    fcstep=cycle,
                    step=(cycle + 1) * n_out * timestep,
                    date=cycle_end_date,
                )
                if decoder_forcings:
                    predict_kwargs["decoder_forcings"] = decoder_forcings

                amp_ctx = torch.autocast(device_type=self.device.type, dtype=self.autocast)
                with torch.inference_mode(), amp_ctx:
                    y_preds = self.predict_step(self.model, input_tensors_torch, **predict_kwargs)

                # Load observations for this cycle and blend. A single input state at
                # the cycle end date carries multi_step_input rows spaced by timestep,
                # which covers the last min(n_out, multi_step_input) predicted steps
                # blended in _da_blend.
                obs_per_dataset = self._load_observations(cycle_end_date, da_states)
                for ds, obs in obs_per_dataset.items():
                    LOG.info("[%s]  Loaded %d observation fields: %s", ds, len(obs), list(obs.keys()))

                input_tensors_torch = self._da_blend(input_tensors_torch, y_preds, obs_per_dataset)

                # Refresh dynamic forcings at the new window dates so any non-observed
                # forcings (e.g. solar geometry from computed providers) are correct
                # for the next cycle / forecast input. The handler fills the last
                # min(multi_step_output, multi_step_input) slots from the last dates.
                for ds, handler in self.tensor_handlers.items():
                    check = constant_check_template[ds].copy()
                    check[handler.metadata.prognostic_input_mask] = True
                    input_tensors_torch[ds] = handler.add_dynamic_forcings_to_input_tensor(
                        input_tensors_torch[ds],
                        da_states[ds],
                        da_target_dates,
                        check,
                    )

                    # Every input column must have been refreshed by the blend or by a
                    # forcing provider. Without this a forcing with no provider would
                    # silently keep its value from the pre-DA input across every cycle.
                    # Mirrors the completeness check in the base forecast loop.
                    if not check.all():
                        mapping = {v: k for k, v in handler.metadata.variable_to_input_tensor_index.items()}
                        missing = [mapping[i] for i in range(check.shape[-1]) if not check[i]]
                        raise ValueError(
                            f"[{ds}] Missing variables in input tensor after DA cycle "
                            f"{cycle + 1}/{self.da_cycles}: {sorted(missing)}"
                        )

                del y_preds

            # ── Update states to reflect DA cycling advance ───────────
            # The input tensors now represent the analysis at
            # start_date + da_cycles*multi_step_output*timestep
            # (= the user's requested forecast reference date).
            analysis_date = start_date + self.da_cycles * n_out * timestep
            analysis_states: dict[str, State] = {}
            for ds in self.dataset_names:
                s = input_states[ds].copy()
                s["date"] = analysis_date
                analysis_states[ds] = s

            LOG.info("DA cycling complete. Analysis date: %s", analysis_date)

            # Convert back to numpy shape (multi_step, features, grid) for the parent's forecast()
            analysis_tensors_numpy: dict[str, FloatArray] = {
                ds: np.swapaxes(input_tensors_torch[ds][0].cpu().numpy(), -2, -1) for ds in self.dataset_names
            }

        # Hook for subclasses at the DA -> forecast transition (e.g. the ensemble
        # runner re-seeds here so members share the analysis but diverge in the
        # forecast). No-op by default.
        self._on_analysis_ready()

        yield from super().forecast(lead_time, analysis_tensors_numpy, analysis_states)

    def _on_analysis_ready(self) -> None:
        """Called once per forecast, after DA cycling completes and before the rollout."""
