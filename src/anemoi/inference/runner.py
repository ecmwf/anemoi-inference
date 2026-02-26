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
import os
import sys
import warnings
from collections.abc import Generator
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any
from typing import Union

import numpy as np
from anemoi.transform.variables.variables import VariableFromMarsVocabulary
from anemoi.utils.config import DotDict
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from anemoi.utils.timer import Timer
from pydantic import BaseModel
from pydantic import ConfigDict

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.device import get_available_device
from anemoi.inference.forcings import Forcings
from anemoi.inference.input import Input
from anemoi.inference.inputs import create_input
from anemoi.inference.lazy import torch
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

        self._checkpoint = classes.checkpoint(config.checkpoint, patch_metadata=config.patch_metadata)
        self.variables = Variables(self)

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
        self.multi_step_input = self.checkpoint.multi_step_input
        self.preload_checkpoint = config.preload_checkpoint
        self.preload_buffer_size = config.preload_buffer_size
        self.precision = config.precision
        self.reference_date = config.date if hasattr(config, "date") else None

        # I/O objects and processors
        self.prognostics_input = self.create_prognostics_input()
        self.constant_forcings_input = self.create_constant_coupled_forcings_input()
        self.dynamic_forcings_input = self.create_dynamic_forcings_input()
        self.boundary_forcings_input = self.create_boundary_forcings_input()

        self.pre_processors = self.create_pre_processors()
        self.post_processors = self.create_post_processors()

        # metadata and tensor handlers for each dataset in the checkpoint
        multi_metadata = self._checkpoint.get_multi_dataset_metadata()
        self.dataset_names = multi_metadata.keys()
        self.tensor_handlers = [
            classes.tensor_handler(
                self,
                metadata=metadata,
                device=self.device,
                allow_nans=self.allow_nans,
                constant_forcings_input=self.constant_forcings_input,
                dynamic_forcings_input=self.dynamic_forcings_input,
                boundary_forcings_input=self.boundary_forcings_input,
            )
            for metadata in multi_metadata.values()
        ]

        if self.verbosity > 2:
            logging.basicConfig(level=logging.DEBUG)
            for logger_name in logging.root.manager.loggerDict:
                logging.getLogger(logger_name).setLevel(logging.DEBUG)

            self.checkpoint.print_indices()

        if self.verbosity > 1:
            from rich.console import Console
            from rich.table import Table

            console = Console(file=sys.stderr)
            table = Table(title="Variable categories")
            table.add_column("Variable", no_wrap=True)
            table.add_column("Categories", no_wrap=True)

            for name, categories in self.checkpoint.variable_categories().items():
                table.add_row(name, ", ".join(categories))

            console.print(table)

        if trace_path := config.trace_path:
            # TODO: multi-dataset support for trace
            from .trace import Trace

            self.trace = Trace(trace_path)
        else:
            self.trace = None

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
        self, *, input_state: State, lead_time: str | int | datetime.timedelta, return_numpy: bool = True
    ) -> Generator[State, None, None]:
        """Run the model.

        Parameters
        ----------
        input_state : State
            The input state.
        lead_time : Union[str, int, datetime.timedelta]
            The lead time.
        return_numpy : bool, optional
            Whether to return the output state fields as numpy arrays, by default True.
            Otherwise, it will return torch tensors.

        Returns
        -------
        Generator[State, None, None]
            The forecasted state.
        """
        # Shallow copy to avoid modifying the user's input state
        input_state = input_state.copy()
        for name in self.dataset_names:
            input_state[name]["fields"] = input_state[name]["fields"].copy()
            LOG.info("-" * 80)
            LOG.info(f"Input state `{name}`:")
            LOG.info(f"  {list(input_state[name]['fields'].keys())}")

        if self.reference_date is None:
            self.reference_date = input_state[self.dataset_names[0]]["date"]

        lead_time = to_timedelta(lead_time)

        with ProfilingRunner(self.use_profiler):
            with ProfilingLabel("Prepare input tensor", self.use_profiler):
                input_tensors = {
                    handler.name: handler.prepare_input_tensor(input_state[handler.name])
                    for handler in self.tensor_handlers
                }

            try:
                yield from self.prepare_output_state(self.forecast(lead_time, input_tensors, input_state), return_numpy)
            except (TypeError, ModuleNotFoundError, AttributeError):
                self.checkpoint.report_error()
                raise
            finally:
                self.complete_forecast_hook()

    def initial_constant_forcings_inputs(self, constant_forcings_inputs: list[Forcings]) -> list[Forcings]:
        """Modify the constant forcings inputs for the first step.

        Parameters
        ----------
        constant_forcings_inputs : list of Forcings
            The constant forcings inputs.

        Returns
        -------
        list[Forcings]
            The modified constant forcings inputs.
        """
        # Give an opportunity to modify the forcings for the first step
        return constant_forcings_inputs

    def initial_dynamic_forcings_inputs(self, dynamic_forcings_inputs: list[Forcings]) -> list[Forcings]:
        """Modify the dynamic forcings inputs for the initial step of the inference process.

        This method provides a hook to adjust the list of dynamic forcings before the first
        inference step is executed. By default, it returns the inputs unchanged, but subclasses
        can override this method to implement custom preprocessing or initialization logic.

        Parameters
        ----------
        dynamic_forcings_inputs : List[Forcings]
            The dynamic forcings inputs to be potentially modified for the initial step.

        Returns
        -------

        List[Forcings]
            The modified list of dynamic forcings inputs for the initial step.

        """
        # Give an opportunity to modify the forcings for the first step
        return dynamic_forcings_inputs

    def prepare_output_state(
        self, output: Generator[State, None, None], return_numpy: bool
    ) -> Generator[State, None, None]:
        """Prepare the output state.

        Parameters
        ----------
        output : Generator[State, None, None]
            Output state generator,
            Expects fields to be torch tensors with shape (values, variables).
        return_numpy : bool
            Whether to return the output state fields as numpy arrays.

        Yields
        ------
        Generator[State, None, None]
            The prepared output state.
        """

        for state in output:
            if return_numpy:
                # Convert fields to numpy arrays
                for handler in self.tensor_handlers:
                    for name, field in state[handler.name]["fields"].items():
                        if isinstance(field, torch.Tensor):
                            state[handler.name]["fields"][name] = field.cpu().numpy()
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
        self, model: "torch.nn.Module", input_tensor_torch: "torch.Tensor", **kwargs: Any
    ) -> "torch.Tensor":
        """Predict the next step.

        Parameters
        ----------
        model : torch.nn.Module
            The model.
        input_tensor_torch : torch.Tensor
            The input tensor.
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
        try:
            # NOTE: This is a temporary hack to support the single dataset case for multi-dataset checkpoints
            # TODO: change when we have a proper multi-dataset runner
            if self.checkpoint._metadata.multi_dataset:
                return model.predict_step({self.checkpoint._metadata.name: input_tensor_torch}, **kwargs)[
                    self.checkpoint._metadata.name
                ]

            return model.predict_step(input_tensor_torch, **kwargs)
        except TypeError:
            # This is for backward compatibility because old models did not
            # have kwargs in the forward or predict_step
            return model.predict_step(input_tensor_torch)

    def forecast_stepper(
        self, start_date: datetime.datetime, lead_time: datetime.timedelta
    ) -> Generator[tuple[datetime.timedelta, datetime.datetime, datetime.datetime, bool], None, None]:
        """Generate step and date variables for the forecast loop.

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
        valid_date : datetime.datetime
            Date of the forecast
        next_date : datetime.datetime
            Date used to prepare the next input tensor
        is_last_step : bool
            True if it's the last step of the forecast
        """
        steps = lead_time // self.checkpoint.timestep

        LOG.info("Lead time: %s, time stepping: %s Forecasting %s steps", lead_time, self.checkpoint.timestep, steps)

        for s in range(steps):
            step = (s + 1) * self.checkpoint.timestep
            valid_date = start_date + step
            next_date = valid_date
            is_last_step = s == steps - 1
            yield step, valid_date, next_date, is_last_step

    def forecast(
        self, lead_time: str, input_tensors_numpy: dict[str, FloatArray], input_state: State
    ) -> Generator[State, None, None]:
        """Forecast the future states.

        Parameters
        ----------
        lead_time : str
            The lead time.
        input_tensors_numpy : dict[str, FloatArray]
            The input tensors.
        input_state : State
            The input state.

        Returns
        -------
        Any
            The forecasted state.
        """
        # NOTE we are not using decorator of the top level function as we anticipate lazy torch load
        with torch.inference_mode():
            self.model.eval()

            # Create pytorch input tensor dict
            input_tensors_torch = {
                name: torch.from_numpy(np.swapaxes(input_tensor_numpy, -2, -1)[np.newaxis, ...]).to(self.device)
                for name, input_tensor_numpy in input_tensors_numpy.items()
            }

            lead_time = to_timedelta(lead_time)

            new_state = input_state.copy()  # We should not modify the input state

            # The variable `check` is used to keep track of which variables have been updated
            # In the input tensor. `reset` is used to reset `check` to False except
            # when the values are of the constant in time variables
            check = {}
            reset = {}
            for handler in self.tensor_handlers:
                new_state[handler.name]["fields"] = dict()
                new_state[handler.name]["step"] = to_timedelta(0)
                start = input_state[handler.name]["date"]

                reset[handler.name] = np.full((input_tensors_torch[handler.name].shape[-1],), False)
                variable_to_input_tensor_index = handler.metadata.variable_to_input_tensor_index
                typed_variables = handler.metadata.typed_variables
                for variable, i in variable_to_input_tensor_index.items():
                    if typed_variables[variable].is_constant_in_time:
                        reset[handler.name][i] = True

                check[handler.name] = reset[handler.name].copy()

            if self.verbosity > 0:
                handler._print_input_tensor("First input tensor", input_tensors_torch)

            for s, (step, date, next_date, is_last_step) in enumerate(self.forecast_stepper(start, lead_time)):
                title = f"Forecasting step {step} ({date})"

                for name in self.dataset_names:
                    new_state[name]["date"] = date
                    new_state[name]["previous_step"] = new_state[name].get("step")
                    new_state[name]["step"] = step

                # if self.trace:
                #     self.trace.write_input_tensor(
                #         date,
                #         s,
                #         input_tensor_torch.cpu().numpy(),
                #         variable_to_input_tensor_index,
                #         self.checkpoint.timestep,
                #     )
                amp_ctx = torch.autocast(device_type=self.device.type, dtype=self.autocast)

                # Predict next state of atmosphere
                with torch.inference_mode(), amp_ctx, ProfilingLabel("Predict step", self.use_profiler), Timer(title):
                    y_pred = self.predict_step(self.model, input_tensors_torch, fcstep=s, step=step, date=date)

                output = {
                    name: torch.squeeze(tensor, dim=(0, 1)) for name, tensor in y_pred.items()
                }  # shape: (values, variables)

                # Update state
                with ProfilingLabel("Updating state (CPU)", self.use_profiler):
                    for handler in self.tensor_handlers:
                        for i in range(output[handler.name].shape[-1]):
                            new_state[handler.name]["fields"][handler.metadata.output_tensor_index_to_variable[i]] = (
                                output[handler.name][..., i].squeeze()
                            )

                        if (s == 0 and self.verbosity > 0) or self.verbosity > 1:
                            handler._print_output_tensor(
                                f"Output tensor - dataset: `{name}`", output[handler.name].cpu().numpy()
                            )

                # if self.trace:
                #     self.trace.write_output_tensor(
                #         date,
                #         s,
                #         output.cpu().numpy(),
                #         self.checkpoint.output_tensor_index_to_variable,
                #         self.checkpoint.timestep,
                #     )

                yield new_state

                # No need to prepare next input tensor if we are at the last step
                if is_last_step:
                    break

                self.output_state_hook(new_state)

                # Update  tensor for next iteration
                with ProfilingLabel("Update tensor for next step", self.use_profiler):
                    for handler in self.tensor_handlers:
                        name = handler.name
                        check[name][:] = reset[name]
                        # if self.trace:
                        #     self.trace.reset_sources(reset[name], self.checkpoint.variable_to_input_tensor_index)

                        input_tensors_torch[name] = handler.copy_prognostic_fields_to_input_tensor(
                            input_tensors_torch[name], y_pred[name], check[name]
                        )

                        del y_pred  # Recover memory

                        input_tensors_torch[name] = handler.add_dynamic_forcings_to_input_tensor(
                            input_tensors_torch[name], new_state[name], next_date, check[name]
                        )
                        input_tensors_torch[name] = handler.add_boundary_forcings_to_input_tensor(
                            input_tensors_torch[name], new_state[name], next_date, check[name]
                        )

                        if not check[name].all():
                            # Not all variables have been updated
                            missing = []
                            variable_to_input_tensor_index = handler.metadata.variable_to_input_tensor_index
                            mapping = {v: k for k, v in variable_to_input_tensor_index.items()}
                            for i in range(check[name].shape[-1]):
                                if not check[name][i]:
                                    missing.append(mapping[i])

                            raise ValueError(f"Missing variables in input tensor: {sorted(missing)}")

                        if (s == 0 and self.verbosity > 0) or self.verbosity > 1:
                            handler._print_input_tensor("Next input tensor", input_tensors_torch)

    def patch_data_request(self, request: Any) -> Any:
        """Patch the data request.

        Parameters
        ----------
        request : Any
            The data request.

        Returns
        -------
        Any
            The patched data request.
        """
        for p in self.pre_processors:
            request = p.patch_data_request(request)

        for p in self.post_processors:
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

        output = self.create_output()

        # In case the constant forcings are from another input, combine them here
        # So that they are in considered in the `write_initial_state`

        prognostic_state = self.prognostics_input.create_input_state(date=self.config.date)
        self._check_state(prognostic_state, "prognostics")

        constants_state = self.constant_forcings_input.create_input_state(date=self.config.date)
        self._check_state(constants_state, "constant_forcings")

        forcings_state = self.dynamic_forcings_input.create_input_state(date=self.config.date)
        self._check_state(forcings_state, "dynamic_forcings")

        input_state = dict(
            data=self._combine_states(
                prognostic_state,
                constants_state,
                forcings_state,
            )
        )

        # This hook is needed for the coupled runner
        self.input_state_hook(constants_state)

        # For step-zero only
        initial_state = Output.reduce(
            self._initial_state(
                prognostic_state,
                constants_state,
                forcings_state,
            )
        )
        # Top-level post-processors on the other hand are applied on State and are executed here.
        LOG.info("Top-level post-processors: %s", self.post_processors)

        for processor in self.post_processors:
            initial_state = processor.process(initial_state)

        output.open(initial_state)

        LOG.info("write_initial_state: %s", output)
        output.write_initial_state(initial_state)

        for state in self.run(input_state=input_state, lead_time=lead_time):
            # Apply top-level post-processors
            for processor in self.post_processors:
                state["data"] = processor.process(state["data"])
            output.write_state(state["data"])

        output.close()

        if "accumulate_from_start_of_forecast" not in self.config.post_processors:
            LOG.warning("""
                🚧 The default accumulation behaviour has changed. 🚧
                🚧 Accumulation fields have NOT been accumulated from the beginning of the forecast. 🚧
                🚧 To accumulate from the beginning, set `post_processors: [accumulate_from_start_of_forecast]` 🚧
                """)  # ecmwf/anemoi-inference#131

    def create_output(self) -> Output:
        """Create the output.

        Returns
        -------
        Output
            The created output.
        """
        output = create_output(self, self.config.output)
        LOG.info("Output: %s", output)
        return output

    def _input_forcings(self, *names: str) -> dict[str, Any]:
        """Get the input forcings configuration.

        Parameters
        ----------
        names : list
            The name of the forcings configuration.

        Returns
        -------
        dict
            The input forcings configuration.
        """

        for name in names:
            if name.startswith("-"):
                deprecated = True
                name = name[1:]  # Remove the leading dash
            else:
                deprecated = False
            if self.config.get(name):
                if deprecated:
                    LOG.warning(
                        f"🚫 The `{name}` input forcings configuration is deprecated. "
                        f"Please use the `{names[0]}` configuration instead."
                    )
                if name != names[0]:
                    LOG.info(f"Loading `config.{names[0]}` from `config.{name}`")

                return self.config[name]

        return self.config.input

    #########################################################################################################
    def create_prognostics_input(self) -> Input:
        """Create the prognostics input.

        Returns
        -------
        Input
            The created prognostics input.
        """
        variables = self.variables.retrieved_prognostic_variables()
        config = self._input_forcings("prognostic_input", "input") if variables else "empty"
        input = create_input(self, config, variables=variables, purpose="prognostics")
        LOG.info("Prognostic input: %s", input)
        return input

    def create_constant_coupled_forcings_input(self) -> Input:
        """Create the constant coupled forcings input.

        Returns
        -------
        Input
            The created constant coupled forcings input.
        """
        variables = self.variables.retrieved_constant_forcings_variables()
        config = self._input_forcings("constant_forcings", "forcings", "input") if variables else "empty"
        input = create_input(self, config, variables=variables, purpose="constant_forcings")
        LOG.info("Constant coupled forcings input: %s", input)
        return input

    def create_dynamic_forcings_input(self) -> Input:
        """Create the dynamic forcings input.

        Returns
        -------
        Input
            The created dynamic forcings input.
        """
        variables = self.variables.retrieved_dynamic_forcings_variables()
        config = self._input_forcings("dynamic_forcings", "-forcings", "input") if variables else "empty"
        input = create_input(self, config, variables=variables, purpose="dynamic_forcings")
        LOG.info("Dynamic forcings input: %s", input)
        return input

    def create_boundary_forcings_input(self) -> Input:
        """Create the boundary forcings input.

        Returns
        -------
        Input
            The created boundary forcings input.
        """
        variables = self.variables.retrieved_prognostic_variables()
        config = self._input_forcings("boundary_forcings", "-boundary", "forcings", "input") if variables else "empty"
        input = create_input(self, config, variables=variables, purpose="boundary_forcings")
        LOG.info("Boundary forcings input: %s", input)
        return input

    def create_pre_processors(self) -> list[Processor]:
        """Create pre-processors.

        Returns
        -------
        List[Processor]
            The created pre-processors.
        """
        result = []
        for processor in self.config.pre_processors:
            result.append(create_pre_processor(self, processor))

        LOG.info("Pre processors: %s", result)
        return result

    def create_post_processors(self) -> list[Processor]:
        """Create post-processors.

        Returns
        -------
        List[Processor]
            The created post-processors.
        """
        result = []
        for processor in self.config.post_processors:
            result.append(create_post_processor(self, processor))

        LOG.info("Post processors: %s", result)
        return result

    def _combine_states(self, *states: dict[str, Any]) -> dict[str, Any]:
        """Combine multiple states into one.

        Parameters
        ----------
        states : list
            The states to combine.

        Returns
        -------
        dict
            The combined state.
        """
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
