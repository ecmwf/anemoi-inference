# (C) Copyright 2024-2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
import os
import sys
from collections.abc import Generator
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any
from typing import Union

import numpy as np
from anemoi.transform.variables.variables import VariableFromMarsVocabulary
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from anemoi.utils.timer import Timer

from anemoi.inference.device import get_available_device
from anemoi.inference.forcings import Forcings
from anemoi.inference.lazy import torch
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


class Kind:
    """Used for debugging purposes."""

    def __init__(self, **kwargs: Any) -> None:
        """Parameters
        -------------
        **kwargs : Any
            Keyword arguments representing the kind attributes.
        """
        self.kwargs = kwargs

    def __repr__(self) -> str:
        """Returns
        ----------
        str
            String representation of the kind.
        """
        result = []

        for k, v in self.kwargs.items():
            if v:
                result.append(k)

        if not result:
            return "?"

        return ", ".join(result)


class Runner(Context):
    """A runner is responsible for running a model."""

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str | None = None,
        precision: str | None = None,
        allow_nans: bool | None = None,
        use_grib_paramid: bool = False,
        verbosity: int = 0,
        patch_metadata: dict[str, Any] = {},
        development_hacks: dict[str, Any] = {},
        trace_path: str | None = None,
        output_frequency: str | None = None,
        write_initial_state: bool = True,
        initial_state_categories: list[str] | None = None,
        use_profiler: bool = False,
        typed_variables: dict[str, dict] = {},
        preload_checkpoint: bool = False,
        preload_buffer_size: int = 32 * 1024 * 1024,
        tensor_handler_class: type[TensorHandler] = TensorHandler,
    ) -> None:
        """Parameters
        -------------
        checkpoint : str
            Path to the checkpoint file.
        device : str | None, optional
            Device to run the model on, by default None.
            If None the device will be automatically detected using :func:`anemoi.inference.device.get_available_device`.
        precision : Optional[str], optional
            Precision to use, by default None.
        allow_nans : Optional[bool], optional
            Whether to allow NaNs, by default None.
        use_grib_paramid : bool, optional
            Whether to use GRIB paramid, by default False.
        verbosity : int, optional
            Verbosity level, by default 0.
        patch_metadata : dict, optional
            Metadata for patching, by default {}.
        development_hacks : dict, optional
            Development hacks, by default {}.
        trace_path : Optional[str], optional
            Path for tracing, by default None.
        output_frequency : Optional[str], optional
            Frequency of output, by default None.
        write_initial_state : bool, optional
            Whether to write the initial state, by default True.
        use_profiler : bool, optional
            Whether to use profiler, by default False.
        preload_checkpoint : bool
            Whether to read the checkpoint file from disk before loading the model, by default False.
        preload_buffer_size : int
            Size of the buffer to use when preloading the checkpoint file, in bytes. Default is 32 MB.
        """
        self._checkpoint = Checkpoint(checkpoint, patch_metadata=patch_metadata)
        self.variables = Variables(self)
        self.trace_path = trace_path

        if trace_path:
            # TODO: multi-dataset support for trace
            from .trace import Trace

            self.trace = Trace(trace_path)
        else:
            self.trace = None

        self._device = device
        self.precision = precision

        multi_metadata = self._checkpoint.get_multi_dataset_metadata()
        self.dataset_names = multi_metadata.keys()
        self.tensor_handlers = [
            tensor_handler_class(self, metadata=metadata, device=self.device) for metadata in multi_metadata.values()
        ]

        # Override the default values set in `Context`
        self.verbosity = verbosity
        self.allow_nans = allow_nans
        self.use_grib_paramid = use_grib_paramid
        self.development_hacks = development_hacks
        self.hacks = bool(development_hacks)
        self.output_frequency = output_frequency
        self.write_initial_state = write_initial_state
        self.initial_state_categories = initial_state_categories
        self.use_profiler = use_profiler

        # For the moment, until we have a better solution
        self.typed_variables = {k: VariableFromMarsVocabulary(k, v) for k, v in typed_variables.items()}

        self._input_kinds = {}
        self._input_tensor_by_name = []

        self._output_kinds = {}
        self._output_tensor_by_name = []

        self.multi_step_input = self.checkpoint.multi_step_input

        self.pre_processors = self.create_pre_processors()
        self.post_processors = self.create_post_processors()
        self.preload_checkpoint = preload_checkpoint
        self.preload_buffer_size = preload_buffer_size

        if self.verbosity > 2:
            logging.basicConfig(level=logging.DEBUG)
            for logger_name in logging.root.manager.loggerDict:
                logging.getLogger(logger_name).setLevel(logging.DEBUG)

            self.checkpoint.print_indices()

        LOG.info("Using %s runner, device=%s", self.__class__.__name__, self.device)

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

    def validate_input_state(self, input_state: State) -> State:
        """Validate the input state.

        Parameters
        ----------
        input_state : State
            The input state.

        Returns
        -------
        State
            The validated input state.
        """
        if not isinstance(input_state, dict):
            raise ValueError("Input state must be a dictionnary")

        EXPECT = dict(date=datetime.datetime, latitudes=np.ndarray, longitudes=np.ndarray, fields=dict)

        for key, klass in EXPECT.items():
            if key not in input_state:
                raise ValueError(f"Input state must contain a `{key}` entry")

            if not isinstance(input_state[key], klass):
                raise ValueError(
                    f"Input state entry `{key}` is type {type(input_state[key])}, expected {klass} instead"
                )

        # Detach from the user's input so we can modify it
        input_state = input_state.copy()
        fields = input_state["fields"] = input_state["fields"].copy()
        number_of_grid_points = self.checkpoint.number_of_grid_points

        for latlon in ("latitudes", "longitudes"):
            if len(input_state[latlon].shape) != 1:
                raise ValueError(f"Input state entry `{latlon}` must be 1D, shape is {input_state[latlon].shape}")

        nlat = len(input_state["latitudes"])
        nlon = len(input_state["longitudes"])
        if nlat != nlon:
            raise ValueError(f"Size mismatch latitudes={nlat}, longitudes={nlon}")

        if nlat != number_of_grid_points:
            raise ValueError(f"Size mismatch latitudes={nlat}, number_of_grid_points={number_of_grid_points}")

        multi_step = self.multi_step_input

        expected_shape = (multi_step, number_of_grid_points)

        LOG.info("Expected shape for each input fields: %s", expected_shape)

        # Check field
        with_nans = []

        for name, field in list(fields.items()):
            # Allow for 1D fields if multi_step is 1
            if len(field.shape) == 1:
                field = fields[name] = field.reshape(1, field.shape[0])

            if field.shape != expected_shape:
                raise ValueError(f"Field `{name}` has the wrong shape. Expected {expected_shape}, got {field.shape}")

            if np.isinf(field).any():
                raise ValueError(f"Field `{name}` contains infinities")

            if np.isnan(field).any():
                with_nans.append(name)

        if with_nans:
            msg = f"NaNs found in the following variables: {sorted(with_nans)}"
            if self.allow_nans is None:
                LOG.warning(msg)
                self.allow_nans = True

            if not self.allow_nans:
                raise ValueError(msg)

        return input_state

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
