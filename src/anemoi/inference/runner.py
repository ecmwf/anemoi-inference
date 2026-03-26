# (C) Copyright 2024-2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import itertools
import logging
import math
import os
import sys
import warnings
from collections.abc import Generator
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import Union

import numpy as np
from anemoi.transform.variables.variables import VariableFromMarsVocabulary
from anemoi.utils.config import DotDict
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from anemoi.utils.timer import Timer
from pydantic import BaseModel
from pydantic import ConfigDict

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.config.utils import input_types_config
from anemoi.inference.config.utils import multi_datasets_config
from anemoi.inference.device import get_available_device
from anemoi.inference.forcings import Forcings
from anemoi.inference.input import Input
from anemoi.inference.inputs import create_input
from anemoi.inference.lazy import torch
from anemoi.inference.metadata import Metadata
from anemoi.inference.output import Output
from anemoi.inference.outputs import create_output
from anemoi.inference.post_processors import create_post_processor
from anemoi.inference.pre_processors import create_pre_processor
from anemoi.inference.processor import Processor
from anemoi.inference.tensors import TensorHandler
from anemoi.inference.types import FloatArray
from anemoi.inference.types import State

from .checkpoint import Checkpoint
from .context import Context
from .precisions import PRECISIONS
from .profiler import ProfilingLabel
from .profiler import ProfilingRunner
from .variables import Variables

if TYPE_CHECKING:
    from anemoi.inference.runners.parallel import ParallelRunnerMixin

LOG = logging.getLogger(__name__)


class RunnerClasses(BaseModel):
    """Configurable class types used by the Runner.
    Child runners can override these with different classes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tensor_handler: type[TensorHandler] = TensorHandler
    checkpoint: type[Checkpoint] = Checkpoint
    metadata: type[Metadata] = Metadata


class Runner(Context):
    """A runner is responsible for running a model.
    This class provides the default forecaster implementation with rollout.
    """

    def __init__(self, config: RunConfiguration, *, classes: RunnerClasses | None = None) -> None:
        self._device = config.device
        LOG.info(f"Using {self.__class__.__name__} runner, device={self.device}")

        classes = classes or RunnerClasses()
        self.classes = classes

        config = DotDict(config.model_dump())
        self.config = config

        self._checkpoint = classes.checkpoint(
            config.checkpoint,
            metadata_base=classes.metadata,
            patch_metadata=config.patch_metadata,
        )

        # override the default values set in `Context`
        self.verbosity = config.verbosity
        self.allow_nans = config.allow_nans
        self.use_grib_paramid = config.use_grib_paramid
        self.development_hacks = config.development_hacks
        self.hacks = bool(config.development_hacks)
        self.output_frequency = config.output_frequency
        self.write_initial_state = config.write_initial_state
        self.initial_state_categories = config.initial_state_categories
        self.use_profiler = config.use_profiler

        # other attributes derived from config or metadata
        self.typed_variables = {k: VariableFromMarsVocabulary(k, v) for k, v in config.typed_variables.items()}
        self.preload_checkpoint = config.preload_checkpoint
        self.preload_buffer_size = config.preload_buffer_size
        self.precision = config.precision
        self.reference_date = config.date if hasattr(config, "date") else None

        # processors, I/O and tensor handlers for each dataset in the checkpoint
        self.pre_processors: dict[str, list[Processor]] = {}
        self.post_processors: dict[str, list[Processor]] = {}
        self.tensor_handlers: dict[str, TensorHandler] = {}
        self.prognostics_inputs: dict[str, Input] = {}
        self.constant_forcings_inputs: dict[str, Input] = {}
        self.dynamic_forcings_inputs: dict[str, Input] = {}
        self.boundary_forcings_inputs: dict[str, Input] = {}
        self.outputs: dict[str, Output] = {}

        multi_metadata = self._checkpoint.multi_dataset_metadata

        for dataset, metadata in multi_metadata.items():
            self.pre_processors[dataset] = self.create_pre_processors(dataset, metadata)
            self.post_processors[dataset] = self.create_post_processors(dataset, metadata)
            self.prognostics_inputs[dataset] = self.create_input("prognostics", dataset, metadata)
            self.constant_forcings_inputs[dataset] = self.create_input("constant_forcings", dataset, metadata)
            self.dynamic_forcings_inputs[dataset] = self.create_input("dynamic_forcings", dataset, metadata)
            self.boundary_forcings_inputs[dataset] = self.create_input("boundary_forcings", dataset, metadata)
            self.outputs[dataset] = self.create_output(dataset, metadata)

            self.tensor_handlers[dataset] = classes.tensor_handler(
                self,
                metadata=metadata,
                constant_forcings_input=self.constant_forcings_inputs[dataset],
                dynamic_forcings_input=self.dynamic_forcings_inputs[dataset],
                boundary_forcings_input=self.boundary_forcings_inputs[dataset],
                trace_path=multi_datasets_config(config.trace_path, dataset),
            )

            if self.verbosity > 2:
                logging.basicConfig(level=logging.DEBUG)
                for logger_name in logging.root.manager.loggerDict:
                    logging.getLogger(logger_name).setLevel(logging.DEBUG)

                metadata.print_indices()

            if self.verbosity > 1:
                from rich.console import Console
                from rich.table import Table

                console = Console(file=sys.stderr)
                table = Table(title=f"\[{metadata.dataset_name}] Variable categories")
                table.add_column("Variable", no_wrap=True)
                table.add_column("Categories", no_wrap=True)

                for dataset, categories in metadata.variable_categories().items():
                    table.add_row(dataset, ", ".join(categories))

                console.print(table)

    @property
    def checkpoint(self) -> Checkpoint:
        """Returns
        ----------
        Checkpoint
            The checkpoint object.
        """
        return self._checkpoint

    @property
    def device(self) -> "torch.device":
        if self._device is None:
            self._device = get_available_device()
        elif isinstance(self._device, str):
            self._device = torch.device(self._device)
        return self._device

    @device.setter
    def device(self, value: "torch.device | str") -> None:
        """Set the device for the runner."""
        if isinstance(value, str):
            value = torch.device(value)
        self._device = value

    def run(
        self, *, input_states: dict[str, State], lead_time: str | int | datetime.timedelta, return_numpy: bool = True
    ) -> Generator[dict[str, State], None, None]:
        """Run the model.

        Parameters
        ----------
        input_states : dict[str, State]
            The input states for each dataset.
        lead_time : Union[str, int, datetime.timedelta]
            The lead time.
        return_numpy : bool, optional
            Whether to return the output state fields as numpy arrays, by default True.
            Otherwise, it will return torch tensors.

        Returns
        -------
        Generator[dict[str, State], None, None]
            The forecasted states.
        """
        # Shallow copy to avoid modifying the user's input state
        input_states = {dataset: state.copy() for dataset, state in input_states.items()}
        for dataset in input_states:
            input_states[dataset]["fields"] = input_states[dataset]["fields"].copy()
            LOG.info("-" * 80)
            LOG.info(f"[{dataset}] Input state:")
            LOG.info(f"  {list(input_states[dataset]['fields'].keys())}")

        if self.reference_date is None:
            self.reference_date = next(iter(input_states.values()))["date"]

        lead_time = to_timedelta(lead_time)

        with ProfilingRunner(self.use_profiler):
            with ProfilingLabel("Prepare input tensor", self.use_profiler):
                input_tensors = {
                    dataset: handler.prepare_input_tensor(input_states[dataset])
                    for dataset, handler in self.tensor_handlers.items()
                }

            try:
                yield from self.prepare_output_state(
                    self.forecast(lead_time, input_tensors, input_states), return_numpy
                )
            except (TypeError, ModuleNotFoundError, AttributeError):
                self.checkpoint.report_error()
                raise
            finally:
                self.complete_forecast_hook()

    def initial_constant_forcings_inputs(self, constant_forcings_inputs: list[Forcings]) -> list[Forcings]:
        """Modify the constant forcings inputs for the first step."""
        # Give an opportunity to modify the forcings for the first step
        return constant_forcings_inputs

    def initial_dynamic_forcings_inputs(self, dynamic_forcings_inputs: list[Forcings]) -> list[Forcings]:
        """Modify the dynamic forcings inputs for the initial step of the inference process.

        This method provides a hook to adjust the list of dynamic forcings before the first
        inference step is executed. By default, it returns the inputs unchanged, but subclasses
        can override this method to implement custom preprocessing or initialization logic.
        """
        # Give an opportunity to modify the forcings for the first step
        return dynamic_forcings_inputs

    def prepare_output_state(
        self, output: Generator[dict[str, State], None, None], return_numpy: bool
    ) -> Generator[dict[str, State], None, None]:
        """Prepare the output state.

        Parameters
        ----------
        output : Generator[dict[str, State], None, None]
            Output state generator.
            Expects a dictionary of states keyed by dataset name.
            Expects fields in each state to be torch tensors with shape (values, variables).
        return_numpy : bool
            Whether to return the output state fields as numpy arrays.

        Yields
        ------
        Generator[dict[str, State], None, None]
            The prepared output state.
        """

        for state in output:
            if return_numpy:
                # Convert fields to numpy arrays
                for name in state:
                    for field_name, field in state[name]["fields"].items():
                        if isinstance(field, torch.Tensor):
                            state[name]["fields"][field_name] = field.cpu().numpy()
            yield state

    @cached_property
    def autocast(self) -> Union["torch.dtype", str]:
        """The autocast precision."""
        autocast = self.precision

        if autocast is None:
            autocast = self.checkpoint.precision

        if autocast is None:
            LOG.warning("No autocast given, using float16")
            autocast = "16"

        return PRECISIONS.get(autocast, autocast)

    @cached_property
    def model(self) -> "torch.nn.Module":
        """Returns
        ----------
        Any
            The loaded model.
        """
        from anemoi.utils.humanize import bytes_to_human

        try:
            size = os.path.getsize(self.checkpoint.path)
            LOG.info("Checkpoint size: %s", bytes_to_human(size))
        except FileNotFoundError:
            # This happens during testing, with mocked checkpoints
            # If we are not in a testing environment, torch.load will raise
            # the proper error
            size = 0
            LOG.warning("Checkpoint file not found: %s", self.checkpoint.path)

        if self.preload_checkpoint and size > 0:
            with Timer(f"Preloading {self.checkpoint}") as t:
                with open(self.checkpoint.path, "rb") as f:
                    while f.read(self.preload_buffer_size):
                        pass
            LOG.info("Preloading checkpoint: %s/s", bytes_to_human(size / t.elapsed))

        with Timer(f"Loading {self.checkpoint}") as t:
            LOG.info("Device is '%s'", self.device)
            LOG.info("Loading model from %s", self.checkpoint.path)

            try:
                model = torch.load(self.checkpoint.path, map_location=self.device, weights_only=False).to(self.device)
            except RuntimeError:
                # This happens when the no GPU is available
                raise
            except Exception as e:  # Wildcard exception to catch all errors
                validation_result = self.checkpoint.validate_environment(on_difference="return")
                e.add_note("Model failed to load, check the stack trace above this message to find the real error")
                e.add_note("Is your environment valid?:\n" + str(validation_result))
                raise e
            # model.set_inference_options(**self.inference_options)
            assert getattr(model, "runner", None) is None, model.runner

            LOG.info("Loading checkpoint: %s/s", bytes_to_human(size / t.elapsed))

            model.runner = self
            return model

    def predict_step(
        self, model: "torch.nn.Module", input_tensors_torch: dict[str, "torch.Tensor"], **kwargs: Any
    ) -> dict[str, "torch.Tensor"]:
        """Predict the next step.

        Parameters
        ----------
        model : torch.nn.Module
            The model.
        input_tensors_torch : dict[str, torch.Tensor]
            The input tensors for each dataset.
        **kwargs : Any
            Additional keyword arguments that will be passed to the model's predict_step method.

        Returns
        -------
        torch.Tensor
            The predicted step.
        """
        for key, value in self.config.predict_kwargs.items():
            if key in kwargs:
                warnings.warn(
                    f"`predict_kwargs` contains illegal kwarg `{key}`. This kwarg is set by the runner and will be ignored."
                )
                continue
            kwargs[key] = value

        if not self.checkpoint.multi_dataset:
            assert len(input_tensors_torch) == 1, "Expected only one dataset in input tensors"
            name, tensor = next(iter(input_tensors_torch.items()))
            return {name: model.predict_step(tensor, **kwargs)}

        return model.predict_step(input_tensors_torch, **kwargs)

    def forecast_stepper(
        self, start_date: datetime.datetime, lead_time: datetime.timedelta
    ) -> Generator[tuple[datetime.timedelta, list[datetime.datetime], list[datetime.datetime], bool], None, None]:
        """Generate step and date variables for the forecast autoregressive loop.

        Parameters
        ----------
        start_date : datetime.datetime
            Start date of the forecast
        lead_time : datetime.timedelta
            Lead time of the forecast

        Returns
        -------
        step : datetime.timedelta
            Time delta since beginning of forecast
        valid_date : list[datetime.datetime]
            Date of the forecast
        next_date : list[datetime.datetime]
            Date used to prepare the next input tensor
        is_last_step : bool
            True if it's the last step of the forecast
        """
        output_horizon = self.checkpoint.timestep * self.checkpoint.multi_step_output
        steps = math.ceil(lead_time / output_horizon)

        LOG.info(
            "Lead time: %s, time stepping: %s, Forecasting %s steps through %s autoregressive steps of %s prediction(s) each.",
            lead_time,
            self.checkpoint.timestep,
            self.checkpoint.multi_step_output * steps,
            steps,
            self.checkpoint.multi_step_output,
        )

        for s in range(steps):
            step = (s + 1) * output_horizon
            valid_dates = [
                start_date + s * output_horizon + self.checkpoint.timestep * (i + 1)
                for i in range(self.checkpoint.multi_step_output)
            ]
            next_dates = valid_dates
            is_last_step = s == steps - 1
            yield step, valid_dates, next_dates, is_last_step

    def forecast(
        self, lead_time: str, input_tensors_numpy: dict[str, FloatArray], input_states: dict[str, State]
    ) -> Generator[dict[str, State], None, None]:
        """Forecast the future states.

        Parameters
        ----------
        lead_time : str
            The lead time.
        input_tensors_numpy : dict[str, FloatArray]
            The input tensors for each dataset, as numpy arrays with shape (multi_step_input, variables, values).
        input_states : dict[str, State]
            The input states for each dataset.

        Returns
        -------
        dict[str, State]
            The forecasted states for each dataset.
        """
        # NOTE we are not using decorator of the top level function as we anticipate lazy torch load
        with torch.inference_mode():
            self.model.eval()

            # Create pytorch input tensor dict
            input_tensors_torch = {
                dataset: torch.from_numpy(np.swapaxes(input_tensor_numpy, -2, -1)[np.newaxis, ...]).to(self.device)
                for dataset, input_tensor_numpy in input_tensors_numpy.items()
            }

            lead_time = to_timedelta(lead_time)

            new_states = input_states.copy()  # We should not modify the input state

            # The variable `check` is used to keep track of which variables have been updated
            # In the input tensor. `reset` is used to reset `check` to False except
            # when the values are of the constant in time variables
            check = {}
            reset = {}
            for dataset, handler in self.tensor_handlers.items():
                new_states[dataset]["fields"] = dict()
                new_states[dataset]["step"] = to_timedelta(0)
                start = input_states[dataset]["date"]

                reset[dataset] = np.full((input_tensors_torch[dataset].shape[-1],), False)
                variable_to_input_tensor_index = handler.metadata.variable_to_input_tensor_index
                typed_variables = handler.metadata.typed_variables
                for variable, i in variable_to_input_tensor_index.items():
                    if typed_variables[variable].is_constant_in_time:
                        reset[dataset][i] = True

                check[dataset] = reset[dataset].copy()

                if self.verbosity > 0:
                    handler._print_input_tensor("First input tensor", input_tensors_torch[dataset])

            for s, (step, dates, next_dates, is_last_step) in enumerate(self.forecast_stepper(start, lead_time)):
                dates_str = "("
                for d in dates:
                    dates_str += f"{d}, "
                dates_str = f"{dates_str[:-2]})"
                title = f"Forecasting, model call {s+1}: horizon {step}, freq. {self.checkpoint.timestep} {dates_str}"

                for dataset, handler in self.tensor_handlers.items():
                    if handler.trace:
                        handler.trace.write_input_tensor(
                            dates[-1],
                            s,
                            input_tensors_torch[dataset].cpu().numpy(),
                            handler.metadata.variable_to_input_tensor_index,
                            self.checkpoint.timestep,
                        )
                amp_ctx = torch.autocast(device_type=self.device.type, dtype=self.autocast)

                # Predict next state of atmosphere
                with torch.inference_mode(), amp_ctx, ProfilingLabel("Predict step", self.use_profiler), Timer(title):
                    y_pred = self.predict_step(self.model, input_tensors_torch, fcstep=s, step=step, date=dates[-1])

                # y_pred (batch, [time], ensemble, values, variables) -> outputs (time, values, variables)
                outputs: dict[str, torch.Tensor] = {}
                for dataset, tensor in y_pred.items():
                    ndim = tensor.ndim
                    assert ndim in (
                        4,
                        5,
                    ), f"[{dataset}] Output tensor should have dimensions (batch, [time], ensemble, values, variables), got {tensor.shape}"
                    if ndim == 4:
                        # pre-multistep models output (batch, ensemble, values, variables)
                        # add a time dimension of 1 for backwards compatibility
                        tensor = tensor.unsqueeze(1)

                    outputs[dataset] = torch.squeeze(tensor, dim=(0, 2))  # shape: (time, values, variables)

                for i in range(self.checkpoint.multi_step_output):
                    # Update state
                    with ProfilingLabel("Updating state (CPU)", self.use_profiler):
                        for dataset, handler in self.tensor_handlers.items():
                            new_states[dataset]["date"] = dates[i]
                            new_states[dataset]["previous_step"] = new_states[dataset].get("step")
                            new_states[dataset]["step"] = (
                                step + (1 + i - self.checkpoint.multi_step_output) * self.checkpoint.timestep
                            )

                            output = outputs[dataset][i, ...]  # shape: (values, variables)

                            for j in range(output.shape[1]):
                                new_states[dataset]["fields"][handler.metadata.output_tensor_index_to_variable[j]] = (
                                    output[:, j]
                                )

                            if (s == 0 and self.verbosity > 0) or self.verbosity > 1:
                                handler._print_output_tensor(f"[{dataset}] Output tensor:", output.cpu().numpy())

                            if handler.trace:
                                handler.trace.write_output_tensor(
                                    dates[i],
                                    s,
                                    output.cpu().numpy(),
                                    handler.metadata.output_tensor_index_to_variable,
                                    self.checkpoint.timestep,
                                )

                    # we only need to check the first dataset's step as they should all be the same
                    if next(iter(new_states.values()))["step"] <= lead_time:
                        yield new_states

                # No need to prepare next input tensor if we are at the last autoregressive step
                if is_last_step:
                    break

                self.output_state_hook(new_states)

                # Update  tensor for next iteration
                with ProfilingLabel("Update tensor for next step", self.use_profiler):
                    for dataset, handler in self.tensor_handlers.items():
                        check[dataset][:] = reset[dataset]
                        if handler.trace:
                            handler.trace.reset_sources(reset[dataset], handler.metadata.variable_to_input_tensor_index)

                        input_tensors_torch[dataset] = handler.copy_prognostic_fields_to_input_tensor(
                            input_tensors_torch[dataset], y_pred[dataset], check[dataset]
                        )

                        del y_pred[dataset]  # Recover memory

                        # some forcings use the new_state(s)
                        # ComputedForcings only uses it to get latlons
                        # For CoupledForcings multi-out not yet supported, last state is only state
                        # ConstantForcings irrelevant
                        # BoundaryForcings currently only work from dataset, there load_forcings_state takes state as argument but doesn't use it
                        # so for now ok to simply pass the last of the multi-out new states:

                        input_tensors_torch[dataset] = handler.add_dynamic_forcings_to_input_tensor(
                            input_tensors_torch[dataset], new_states[dataset], next_dates, check[dataset]
                        )
                        input_tensors_torch[dataset] = handler.add_boundary_forcings_to_input_tensor(
                            input_tensors_torch[dataset], new_states[dataset], next_dates, check[dataset]
                        )

                        if not check[dataset].all():
                            # Not all variables have been updated
                            missing = []
                            variable_to_input_tensor_index = handler.metadata.variable_to_input_tensor_index
                            mapping = {v: k for k, v in variable_to_input_tensor_index.items()}
                            for i in range(check[dataset].shape[-1]):
                                if not check[dataset][i]:
                                    missing.append(mapping[i])

                            raise ValueError(f"[{dataset}] Missing variables in input tensor: {sorted(missing)}")

                        if (s == 0 and self.verbosity > 0) or self.verbosity > 1:
                            handler._print_input_tensor(f"[{dataset}] Next input tensor", input_tensors_torch[dataset])

    def patch_data_request(self, request: dict, dataset_name: str) -> dict:
        for p in self.pre_processors[dataset_name]:
            request = p.patch_data_request(request)

        for p in self.post_processors[dataset_name]:
            request = p.patch_data_request(request)

        return request

    def _configure_parallel_runner(self: "ParallelRunnerMixin") -> None:
        """Configure the parallel runner (only applies when using the `parallel` runner).

        This method is called by the parallel runner on initialisation.
        Derived classes can implement this method to modify itself for parallel operation.
        """
        pass

    def input_state_hook(self, input_state: State) -> None:
        """Hook used by coupled runners to send the input state."""
        pass

    def output_state_hook(self, state: State) -> None:
        """Hook used by coupled runners to send the input state."""
        pass

    def complete_forecast_hook(self) -> None:
        """Hook called at the end of the forecast."""
        pass

    def has_split_input(self) -> bool:
        # To be overridden by a subclass if the we use different inputs
        # for initial conditions, constants and dynamic forcings
        raise NotImplementedError(
            "This method should be overridden by a subclass if the runner uses different inputs "
            "for initial conditions, constants and dynamic forcings."
        )

    ###########################################################################################################
    def execute(self) -> None:
        """Execute the runner."""

        if self.config.description is not None:
            LOG.info("%s", self.config.description)

        lead_time = to_timedelta(self.config.lead_time)

        # This may be used by Output objects to compute the step
        self.lead_time = lead_time
        self.time_step = self.checkpoint.timestep

        # In case the constant forcings are from another input, combine them here
        # So that they are in considered in the `write_initial_state`

        # input states for each dataset
        input_states: dict[str, State] = {}
        initial_states: dict[str, State] = {}
        for dataset in self.tensor_handlers:
            prognostic_state = self.prognostics_inputs[dataset].create_input_state(date=self.config.date)
            self._check_state(prognostic_state, "prognostics")

            constants_state = self.constant_forcings_inputs[dataset].create_input_state(date=self.config.date)
            self._check_state(constants_state, "constant_forcings")

            forcings_state = self.dynamic_forcings_inputs[dataset].create_input_state(date=self.config.date)
            self._check_state(forcings_state, "dynamic_forcings")

            input_states[dataset] = self._combine_states(
                prognostic_state,
                constants_state,
                forcings_state,
            )

            # This hook is needed for the coupled runner
            self.input_state_hook(constants_state)

            # For step-zero only
            initial_states[dataset] = Output.reduce(
                self._initial_state(
                    prognostic_state,
                    constants_state,
                    forcings_state,
                )
            )

            # Top-level post-processors on the other hand are applied on State and are executed here.
            LOG.info(f"[{dataset}] Top-level post-processors: {self.post_processors[dataset]}")

            for processor in self.post_processors[dataset]:
                initial_states[dataset] = processor.process(initial_states[dataset])

        for dataset, state in initial_states.items():
            self.outputs[dataset].open(initial_states[dataset])

            LOG.info(f"[{dataset}] write_initial_state: {self.outputs[dataset]}")
            self.outputs[dataset].write_initial_state(state)

        for states in self.run(input_states=input_states, lead_time=lead_time):
            for dataset, state in states.items():
                # Apply top-level post-processors
                for processor in self.post_processors[dataset]:
                    state = processor.process(state)
                self.outputs[dataset].write_state(state)

        for output in self.outputs.values():
            output.close()

    #########################################################################################################
    def create_output(self, dataset_name: str, metadata: Metadata) -> Output:
        config = multi_datasets_config(self.config.output, dataset_name)
        output = create_output(self, config, metadata)
        LOG.info(f"[{dataset_name}] Output: {output}")
        return output

    def create_input(
        self,
        input_type: Literal["prognostics", "constant_forcings", "dynamic_forcings", "boundary_forcings"],
        dataset_name: str,
        metadata: Metadata,
    ) -> Input:
        variables = Variables(metadata)
        match input_type:
            case "prognostics":
                variables = variables.retrieved_prognostic_variables()
                config = input_types_config(self.config, "prognostic_input", "input") if variables else "empty"
            case "constant_forcings":
                variables = variables.retrieved_constant_forcings_variables()
                config = input_types_config(self.config, input_type, "forcings", "input") if variables else "empty"
            case "dynamic_forcings":
                variables = variables.retrieved_dynamic_forcings_variables()
                config = input_types_config(self.config, input_type, "-forcings", "input") if variables else "empty"
            case "boundary_forcings":
                variables = variables.retrieved_prognostic_variables()
                config = (
                    input_types_config(self.config, input_type, "-boundary", "forcings", "input")
                    if variables
                    else "empty"
                )
            case _:
                raise ValueError(f"Unknown input type: {input_type}")

        config = multi_datasets_config(config, dataset_name)
        input = create_input(self, config, metadata, variables=variables, purpose=input_type)

        LOG.info(f"[{dataset_name}] {input_type.replace('_', ' ').capitalize()} input: {input}")
        return input

    def create_pre_processors(self, dataset_name: str, metadata: Metadata) -> list[Processor]:
        result = []
        config = multi_datasets_config(self.config.pre_processors, dataset_name)
        for processor in config:
            result.append(create_pre_processor(self, processor, metadata))

        LOG.info(f"[{dataset_name}] Pre processors: {result}")
        return result

    def create_post_processors(self, dataset_name: str, metadata: Metadata) -> list[Processor]:
        result = []
        config = multi_datasets_config(self.config.post_processors, dataset_name)
        for processor in config:
            result.append(create_post_processor(self, processor, metadata))

        LOG.info(f"[{dataset_name}] Post processors: {result}")
        return result

    def _combine_states(self, *states: dict[str, Any]) -> dict[str, Any]:
        """Combine multiple states into one."""
        import numpy as np

        combined = states[0].copy()
        combined["fields"] = combined["fields"].copy()
        shape = None
        first_input = combined.get("_input")

        for state in states[1:]:
            this_input = state.get("_input")

            for name, values in itertools.chain(combined["fields"].items(), state.get("fields", {}).items()):
                if shape is None:
                    shape = values.shape
                elif shape != values.shape:
                    raise ValueError(
                        f"Field '{name}' has different shape in the states: "
                        f"{shape} and {values.shape}."
                        f" Input: {first_input} vs {this_input}."
                    )

            if not set(combined["fields"]).isdisjoint(state["fields"]):
                raise ValueError(
                    f"Some states have overlapping fields:"
                    f" {set(combined['fields']).intersection(state['fields'])}"
                    f" Input: {first_input} vs {this_input}."
                )

            combined["fields"].update(state.get("fields", {}))

            for key, value in state.items():
                if key == "fields":
                    continue

                if key.startswith("_"):
                    continue

                if combined.get(key) is None:
                    combined[key] = value
                    continue

                if value is None:
                    continue

                if type(combined[key]) is not type(value):
                    raise ValueError(
                        f"Key '{key}' has different types in the states: " f"{type(combined[key])} and {type(value)}."
                    )

                if isinstance(value, np.ndarray) and isinstance(combined[key], np.ndarray):
                    if not np.array_equal(combined[key], value):
                        raise ValueError(
                            f"Key '{key}' has different array values in the states: "
                            f"{combined[key]} ({combined[key].shape}) and {value} ({value.shape})."
                            f" Input: {first_input} vs {this_input}."
                        )
                    continue

                if combined[key] != value:
                    raise ValueError(
                        f"Key '{key}' has different values in the states: "
                        f"{combined[key]} and {value} ({shape})."
                        f" Input: {first_input} vs {this_input}."
                    )

        return combined

    def _initial_state(
        self, prognostic_state: dict[str, Any], constants_state: dict[str, Any], forcings_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Create the initial state for the output.

        Parameters
        ----------
        prognostic_state : dict
            The prognostic state.
        constants_state : dict
            The constant forcings state.
        forcings_state : dict
            The dynamic forcings state.

        Returns
        -------
        dict
            The initial state for the output.
        """
        states = []

        if "prognostics" in self.config.initial_state_categories:
            states.append(prognostic_state)

        if "constant_forcings" in self.config.initial_state_categories:
            states.append(constants_state)

        if "dynamic_forcings" in self.config.initial_state_categories:
            states.append(forcings_state)

        return self._combine_states(*states)

    def _check_state(self, state: dict[str, Any], title: str) -> None:
        """Check the state for consistency.

        Parameters
        ----------
        state : dict
            The state to check.
        title : str
            The title of the state (for logging).

        Raises
        ------
        ValueError
            If the state is not consistent.
        """

        if not isinstance(state, dict):
            raise ValueError(f"State '{title}' is not a dictionary: {state}")

        input = state.get("_input")

        if "fields" not in state:
            raise ValueError(f"State '{title}' does not contain 'fields': {state} ({input=})")

        shape = None

        for field, values in state["fields"].items():
            if shape is None:
                shape = values.shape
            elif shape != values.shape:
                raise ValueError(
                    f"Field '{field}' in state '{title}' has different shape: "
                    f"{shape} and {values.shape} ({input=})."
                )

        date = state.get("date")
        if date is None and len(state["fields"]) > 0:
            # date can be None for an empty input
            if not isinstance(date, datetime.datetime):
                raise ValueError(f"State '{title}' does not contain 'date', or it is not a datetime: {date} ({input=})")
