# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
import warnings
from collections.abc import Generator
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import torch
from anemoi.transform.variables.variables import VariableFromMarsVocabulary
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from anemoi.utils.text import table
from anemoi.utils.timer import Timer
from numpy.typing import DTypeLike

from anemoi.inference.forcings import Forcings
from anemoi.inference.types import BoolArray
from anemoi.inference.types import FloatArray
from anemoi.inference.types import State

from .checkpoint import Checkpoint
from .context import Context
from .precisions import PRECISIONS
from .profiler import ProfilingLabel
from .profiler import ProfilingRunner

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
        device: str = "cuda",
        precision: str | None = None,
        report_error: bool = False,
        allow_nans: bool | None = None,
        use_grib_paramid: bool = False,
        verbosity: int = 0,
        patch_metadata: dict[str, Any] = {},
        development_hacks: dict[str, Any] = {},
        trace_path: str | None = None,
        output_frequency: str | None = None,
        write_initial_state: bool = True,
        use_profiler: bool = False,
        typed_variables: dict[str, dict] = {},
    ) -> None:
        """Parameters
        -------------
        checkpoint : str
            Path to the checkpoint file.
        device : str, optional
            Device to run the model on, by default "cuda".
        precision : Optional[str], optional
            Precision to use, by default None.
        report_error : bool, optional
            Whether to report errors, by default False.
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
        """
        self._checkpoint = Checkpoint(checkpoint, patch_metadata=patch_metadata)

        self.trace_path = trace_path

        if trace_path:
            from .trace import Trace

            self.trace = Trace(trace_path)
        else:
            self.trace = None

        self.device = device
        self.precision = precision
        self.report_error = report_error

        # Override the default values set in `Context`
        self.verbosity = verbosity
        self.allow_nans = allow_nans
        self.use_grib_paramid = use_grib_paramid
        self.development_hacks = development_hacks
        self.hacks = bool(development_hacks)
        self.output_frequency = output_frequency
        self.write_initial_state = write_initial_state
        self.use_profiler = use_profiler

        # For the moment, until we have a better solution
        self.typed_variables = {k: VariableFromMarsVocabulary(k, v) for k, v in typed_variables.items()}

        self._input_kinds = {}
        self._input_tensor_by_name = []

        self._output_kinds = {}
        self._output_tensor_by_name = []

        self.pre_processors = self.create_pre_processors()
        self.post_processors = self.create_post_processors()

        if self.verbosity > 2:
            logging.basicConfig(level=logging.DEBUG)
            for logger_name in logging.root.manager.loggerDict:
                logging.getLogger(logger_name).setLevel(logging.DEBUG)

            self.checkpoint.print_indices()

        LOG.info("Using %s runner, device=%s", self.__class__.__name__, self.device)

        if self.verbosity > 1:
            self.checkpoint.print_variable_categories()

    @property
    def checkpoint(self) -> Checkpoint:
        """Returns
        ----------
        Checkpoint
            The checkpoint object.
        """
        return self._checkpoint

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
        input_state["fields"] = input_state["fields"].copy()

        self.constant_forcings_inputs = self.create_constant_forcings_inputs(input_state)
        self.dynamic_forcings_inputs = self.create_dynamic_forcings_inputs(input_state)
        self.boundary_forcings_inputs = self.create_boundary_forcings_inputs(input_state)

        LOG.info("-" * 80)
        LOG.info("Input state:")
        LOG.info(f"  {list(input_state['fields'].keys())}")

        LOG.info("Constant forcings inputs:")
        for f in self.constant_forcings_inputs:
            LOG.info(f"  {f}")

        LOG.info("Dynamic forcings inputs:")
        for f in self.dynamic_forcings_inputs:
            LOG.info(f"  {f}")

        LOG.info("Boundary forcings inputs:")
        for f in self.boundary_forcings_inputs:
            LOG.info(f"  {f}")
        LOG.info("-" * 80)

        lead_time = to_timedelta(lead_time)

        with ProfilingRunner(self.use_profiler):
            with ProfilingLabel("Prepare input tensor", self.use_profiler):
                input_tensor = self.prepare_input_tensor(input_state)

            try:
                yield from self.prepare_output_state(self.forecast(lead_time, input_tensor, input_state), return_numpy)
            except (TypeError, ModuleNotFoundError, AttributeError):
                if self.report_error:
                    self.checkpoint.report_error()
                raise

    def create_constant_forcings_inputs(self, input_state: State) -> list[Forcings]:
        """Create constant forcings inputs.

        Parameters
        ----------
        input_state : State
            The input state.

        Returns
        -------
        list[Forcings]
            The created constant forcings inputs.
        """
        return self.checkpoint.constant_forcings_inputs(self, input_state)

    def create_dynamic_forcings_inputs(self, input_state: State) -> list[Forcings]:
        """Create dynamic forcings inputs.

        Parameters
        ----------
        input_state : State
            The input state.

        Returns
        -------
        list[Forcings]
            The created dynamic forcings inputs.
        """
        return self.checkpoint.dynamic_forcings_inputs(self, input_state)

    def create_boundary_forcings_inputs(self, input_state: State) -> list[Forcings]:
        """Create boundary forcings inputs.

        Parameters
        ----------
        input_state : State
            The input state.

        Returns
        -------
        list[Forcings]
            The created boundary forcings inputs.
        """
        return self.checkpoint.boundary_forcings_inputs(self, input_state)

    def add_initial_forcings_to_input_state(self, input_state: State) -> None:
        """Add initial forcings to the input state.

        Parameters
        ----------
        input_state : State
            The input state.
        """
        # Should that be alreay a list of dates
        date = input_state["date"]
        fields = input_state["fields"]

        dates = [date + h for h in self.checkpoint.lagged]

        # For output object. Should be moved elsewhere
        self.reference_date = dates[-1]
        self.initial_dates = dates

        # TODO: Check for user provided forcings

        # We may need different forcings initial conditions
        initial_constant_forcings_inputs = self.initial_constant_forcings_inputs(self.constant_forcings_inputs)
        initial_dynamic_forcings_inputs = self.initial_dynamic_forcings_inputs(self.dynamic_forcings_inputs)

        LOG.info("-" * 80)
        LOG.info("Initial forcings:")
        LOG.info("  Constant forcings inputs:")
        for f in initial_constant_forcings_inputs:
            LOG.info(f"    {f}")
        LOG.info("  Dynamic forcings inputs:")
        for f in initial_dynamic_forcings_inputs:
            LOG.info(f"    {f}")
        LOG.info("-" * 80)

        for source in initial_constant_forcings_inputs:
            LOG.info("Constant forcings input: %s %s (%s)", source, source.variables, dates)
            arrays = source.load_forcings_array(dates, input_state)
            for name, forcing in zip(source.variables, arrays):
                assert isinstance(forcing, np.ndarray), (name, forcing)
                fields[name] = forcing
                self._input_kinds[name] = Kind(forcing=True, constant=True, **source.kinds)
                if self.trace:
                    self.trace.from_source(name, source, "initial constant forcings")

        for source in initial_dynamic_forcings_inputs:
            LOG.info("Dynamic forcings input: %s %s (%s)", source, source.variables, dates)
            arrays = source.load_forcings_array(dates, input_state)
            for name, forcing in zip(source.variables, arrays):
                assert isinstance(forcing, np.ndarray), (name, forcing)
                fields[name] = forcing
                self._input_kinds[name] = Kind(forcing=True, constant=False, **source.kinds)
                if self.trace:
                    self.trace.from_source(name, source, "initial dynamic forcings")

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
        """Modify the dynamic forcings inputs for the first step.

        Parameters
        ----------
        dynamic_forcings_inputs : list of Forcings
            The dynamic forcings inputs.

        Returns
        -------
        list[Forcings]
            The modified dynamic forcings inputs.
        """
        # Give an opportunity to modify the forcings for the first step
        return dynamic_forcings_inputs

    def prepare_input_tensor(self, input_state: State, dtype: DTypeLike = np.float32) -> FloatArray:
        """Prepare the input tensor.

        Parameters
        ----------
        input_state : State
            The input state.
        dtype : type, optional
            The data type, by default np.float32.

        Returns
        -------
        FloatArray
            The prepared input tensor.
        """
        if "latitudes" not in input_state:
            input_state["latitudes"] = self.checkpoint.latitudes

        if "longitudes" not in input_state:
            input_state["longitudes"] = self.checkpoint.longitudes

        if input_state.get("latitudes") is None or input_state.get("longitudes") is None:
            raise ValueError("Input state must contain 'latitudes' and 'longitudes'")

        typed_variables = self.checkpoint.typed_variables

        for name in input_state["fields"]:
            self._input_kinds[name] = Kind(input=True, constant=typed_variables[name].is_constant_in_time)

        # Add initial forcings to input state if needed

        self.add_initial_forcings_to_input_state(input_state)

        input_state = self.validate_input_state(input_state)

        input_fields = input_state["fields"]

        input_tensor_numpy = np.full(
            shape=(
                self.checkpoint.multi_step_input,
                self.checkpoint.number_of_input_features,
                input_state["latitudes"].size,
            ),
            fill_value=np.nan,
            dtype=dtype,
        )

        self._input_tensor_by_name = [None] * self.checkpoint.number_of_input_features

        LOG.info("Preparing input tensor with shape %s", input_tensor_numpy.shape)

        variable_to_input_tensor_index = self.checkpoint.variable_to_input_tensor_index

        check = set()
        for var, field in input_fields.items():
            i = variable_to_input_tensor_index[var]
            if i in check:
                raise ValueError(f"Duplicate variable {var}/{i} in input fields")
            input_tensor_numpy[:, i] = field
            check.add(i)

            self._input_tensor_by_name[i] = var

        if len(check) != self.checkpoint.number_of_input_features:
            missing = set(range(self.checkpoint.number_of_input_features)) - check
            mapping = {v: k for k, v in self.checkpoint.variable_to_input_tensor_index.items()}
            raise ValueError(f"Missing variables in input fields: {[mapping.get(_, _) for _ in missing]}")

        return input_tensor_numpy

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
                for name, field in state["fields"].items():
                    if isinstance(field, torch.Tensor):
                        state["fields"][name] = field.cpu().numpy()
            yield state

    @cached_property
    def autocast(self) -> torch.dtype | str:
        """The autocast precision."""
        autocast = self.precision

        if autocast is None:
            autocast = self.checkpoint.precision

        if autocast is None:
            LOG.warning("No autocast given, using float16")
            autocast = "16"

        return PRECISIONS.get(autocast, autocast)

    @cached_property
    def model(self) -> torch.nn.Module:
        """Returns
        ----------
        Any
            The loaded model.
        """
        with Timer(f"Loading {self.checkpoint}"):
            LOG.info("Device is '%s'", self.device)
            LOG.info("Loading model from %s", self.checkpoint.path)

            try:
                model = torch.load(self.checkpoint.path, map_location=self.device, weights_only=False).to(self.device)
            except Exception as e:  # Wildcard exception to catch all errors
                if self.report_error:
                    self.checkpoint.report_error()
                validation_result = self.checkpoint.validate_environment(on_difference="return")
                e.add_note("Model failed to load, check the stack trace above this message to find the real error")
                e.add_note("Is your environment valid?:\n" + str(validation_result))
                raise e
            # model.set_inference_options(**self.inference_options)
            assert getattr(model, "runner", None) is None, model.runner
            model.runner = self
            return model

    def predict_step(self, model: torch.nn.Module, input_tensor_torch: torch.Tensor, **kwargs: Any) -> torch.Tensor:
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
        self, lead_time: str, input_tensor_numpy: FloatArray, input_state: State
    ) -> Generator[State, None, None]:
        """Forecast the future states.

        Parameters
        ----------
        lead_time : str
            The lead time.
        input_tensor_numpy : FloatArray
            The input tensor.
        input_state : State
            The input state.

        Returns
        -------
        Any
            The forecasted state.
        """
        self.model.eval()

        torch.set_grad_enabled(False)

        # Create pytorch input tensor
        input_tensor_torch = torch.from_numpy(np.swapaxes(input_tensor_numpy, -2, -1)[np.newaxis, ...]).to(self.device)

        lead_time = to_timedelta(lead_time)

        new_state = input_state.copy()  # We should not modify the input state
        new_state["fields"] = dict()
        new_state["step"] = to_timedelta(0)

        start = input_state["date"]

        # The variable `check` is used to keep track of which variables have been updated
        # In the input tensor. `reset` is used to reset `check` to False except
        # when the values are of the constant in time variables

        reset = np.full((input_tensor_torch.shape[-1],), False)
        variable_to_input_tensor_index = self.checkpoint.variable_to_input_tensor_index
        typed_variables = self.checkpoint.typed_variables
        for variable, i in variable_to_input_tensor_index.items():
            if typed_variables[variable].is_constant_in_time:
                reset[i] = True

        check = reset.copy()

        if self.verbosity > 0:
            self._print_input_tensor("First input tensor", input_tensor_torch)

        for s, (step, date, next_date, is_last_step) in enumerate(self.forecast_stepper(start, lead_time)):
            title = f"Forecasting step {step} ({date})"

            new_state["date"] = date
            new_state["previous_step"] = new_state.get("step")
            new_state["step"] = step

            if self.trace:
                self.trace.write_input_tensor(
                    date, s, input_tensor_torch.cpu().numpy(), variable_to_input_tensor_index, self.checkpoint.timestep
                )

            # Predict next state of atmosphere
            with (
                torch.autocast(device_type=self.device, dtype=self.autocast),
                ProfilingLabel("Predict step", self.use_profiler),
                Timer(title),
            ):
                y_pred = self.predict_step(self.model, input_tensor_torch, fcstep=s, step=step, date=date)

            output = torch.squeeze(y_pred)  # shape: (values, variables)

            # Update state
            with ProfilingLabel("Updating state (CPU)", self.use_profiler):
                for i in range(output.shape[1]):
                    new_state["fields"][self.checkpoint.output_tensor_index_to_variable[i]] = output[:, i]

            if (s == 0 and self.verbosity > 0) or self.verbosity > 1:
                self._print_output_tensor("Output tensor", output.cpu().numpy())

            if self.trace:
                self.trace.write_output_tensor(
                    date,
                    s,
                    output.cpu().numpy(),
                    self.checkpoint.output_tensor_index_to_variable,
                    self.checkpoint.timestep,
                )

            yield new_state

            # No need to prepare next input tensor if we are at the last step
            if is_last_step:
                break

            # Update  tensor for next iteration
            with ProfilingLabel("Update tensor for next step", self.use_profiler):
                check[:] = reset
                if self.trace:
                    self.trace.reset_sources(reset, self.checkpoint.variable_to_input_tensor_index)

                input_tensor_torch = self.copy_prognostic_fields_to_input_tensor(input_tensor_torch, y_pred, check)

                del y_pred  # Recover memory

                input_tensor_torch = self.add_dynamic_forcings_to_input_tensor(
                    input_tensor_torch, new_state, next_date, check
                )
                input_tensor_torch = self.add_boundary_forcings_to_input_tensor(
                    input_tensor_torch, new_state, next_date, check
                )

            if not check.all():
                # Not all variables have been updated
                missing = []
                variable_to_input_tensor_index = self.checkpoint.variable_to_input_tensor_index
                mapping = {v: k for k, v in variable_to_input_tensor_index.items()}
                for i in range(check.shape[-1]):
                    if not check[i]:
                        missing.append(mapping[i])

                raise ValueError(f"Missing variables in input tensor: {sorted(missing)}")

            if (s == 0 and self.verbosity > 0) or self.verbosity > 1:
                self._print_input_tensor("Next input tensor", input_tensor_torch)

    def copy_prognostic_fields_to_input_tensor(
        self, input_tensor_torch: torch.Tensor, y_pred: torch.Tensor, check: BoolArray
    ) -> torch.Tensor:
        """Copy prognostic fields to the input tensor.

        Parameters
        ----------
        input_tensor_torch : torch.Tensor
            The input tensor.
        y_pred : torch.Tensor
            The predicted tensor.
        check : BoolArray
            The check array.

        Returns
        -------
        torch.Tensor
            The updated input tensor.
        """
        # input_tensor_torch is shape: (batch, multi_step_input, values, variables)
        # batch is always 1

        prognostic_output_mask = self.checkpoint.prognostic_output_mask
        prognostic_input_mask = self.checkpoint.prognostic_input_mask

        # Copy prognostic fields to input tensor
        prognostic_fields = y_pred[..., prognostic_output_mask]  # Get new predicted values
        input_tensor_torch = input_tensor_torch.roll(-1, dims=1)  # Roll the tensor in the multi_step_input dimension
        input_tensor_torch[:, -1, :, self.checkpoint.prognostic_input_mask] = (
            prognostic_fields  # Add new values to last 'multi_step_input' row
        )

        assert not check[prognostic_input_mask].any()  # Make sure we are not overwriting some values
        check[prognostic_input_mask] = True

        for n in prognostic_input_mask:
            self._input_kinds[self._input_tensor_by_name[n]] = Kind(prognostic=True)
            if self.trace:
                self.trace.from_rollout(self._input_tensor_by_name[n])

        return input_tensor_torch

    def add_dynamic_forcings_to_input_tensor(
        self, input_tensor_torch: torch.Tensor, state: State, date: datetime.datetime, check: BoolArray
    ) -> torch.Tensor:
        """Add dynamic forcings to the input tensor.

        Parameters
        ----------
        input_tensor_torch : torch.Tensor
            The input tensor.
        state : State
            The state.
        date : datetime.datetime
            The date.
        check : BoolArray
            The check array.

        Returns
        -------
        torch.Tensor
            The updated input tensor.
        """
        if self.hacks:
            if "dynamic_forcings_date" in self.development_hacks:
                date = self.development_hacks["dynamic_forcings_date"]
                warnings.warn(f"🧑‍💻 Using `dynamic_forcings_date` hack: {date} 🧑‍💻")

        # TODO: check if there were not already loaded as part of the input state

        # input_tensor_torch is shape: (batch, multi_step_input, values, variables)
        # batch is always 1

        for source in self.dynamic_forcings_inputs:
            forcings = source.load_forcings_array([date], state)  # shape: (variables, dates, values)

            forcings = np.squeeze(forcings, axis=1)  # Drop the dates dimension

            forcings = np.swapaxes(forcings[np.newaxis, np.newaxis, ...], -2, -1)  # shape: (1, 1, values, variables)

            forcings = torch.from_numpy(forcings).to(self.device)  # Copy to device

            input_tensor_torch[:, -1, :, source.mask] = forcings  # Copy forcings to last 'multi_step_input' row

            assert not check[source.mask].any()  # Make sure we are not overwriting some values
            check[source.mask] = True

            for n in source.mask:
                self._input_kinds[self._input_tensor_by_name[n]] = Kind(forcing=True, **source.kinds)

            if self.trace:
                for n in source.mask:
                    self.trace.from_source(self._input_tensor_by_name[n], source, "dynamic forcings")

        return input_tensor_torch

    def add_boundary_forcings_to_input_tensor(
        self, input_tensor_torch: torch.Tensor, state: State, date: datetime.datetime, check: BoolArray
    ) -> torch.Tensor:
        """Add boundary forcings to the input tensor.

        Parameters
        ----------
        input_tensor_torch : torch.Tensor
            The input tensor.
        state : State
            The state.
        date : datetime.datetime
            The date.
        check : BoolArray
            The check array.

        Returns
        -------
        torch.Tensor
            The updated input tensor.
        """
        # input_tensor_torch is shape: (batch, multi_step_input, values, variables)
        # batch is always 1
        sources = self.boundary_forcings_inputs
        for source in sources:
            forcings = source.load_forcings_array([date], state)  # shape: (variables, dates, values)

            forcings = np.squeeze(forcings, axis=1)  # Drop the dates dimension

            forcings = np.swapaxes(forcings[np.newaxis, np.newaxis, ...], -2, -1)  # shape: (1, 1, values, variables)
            forcings = torch.from_numpy(forcings).to(self.device)  # Copy to device
            total_mask = np.ix_([0], [-1], source.spatial_mask, source.variables_mask)
            input_tensor_torch[total_mask] = forcings  # Copy forcings to last 'multi_step_input' row

        # TO DO: add some consistency checks as above
        return input_tensor_torch

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

        multi_step = self.checkpoint.multi_step_input

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

    def _print_tensor(
        self, title: str, tensor_numpy: FloatArray, tensor_by_name: list[str], kinds: dict[str, Kind]
    ) -> None:
        """Print the tensor.

        Parameters
        ----------
        title : str
            The title.
        tensor_numpy : FloatArray
            The tensor.
        tensor_by_name : list
            The tensor by name.
        kinds : dict
            The kinds.
        """
        assert len(tensor_numpy.shape) == 3, tensor_numpy.shape
        assert tensor_numpy.shape[0] in (1, self.checkpoint.multi_step_input), tensor_numpy.shape
        assert tensor_numpy.shape[1] == len(tensor_by_name), tensor_numpy.shape

        t = []
        for k, v in enumerate(tensor_by_name):
            data = tensor_numpy[-1, k]

            nans = "-"

            if np.isnan(data).any():
                nan_count = np.isnan(data).sum()

                ratio = nan_count / data.size
                nans = f"{ratio:.0%}"

            if np.isinf(data).any():
                nans = "∞"

            t.append((k, v, np.nanmin(data), np.nanmax(data), nans, kinds.get(v, Kind())))

        LOG.info("")
        LOG.info(
            "%s:\n\n%s\n", title, table(t, header=["Index", "Variable", "Min", "Max", "NaNs", "Kind"], align="><<<|<")
        )
        LOG.info("")

    def _print_input_tensor(self, title: str, input_tensor_torch: torch.Tensor) -> None:
        """Print the input tensor.

        Parameters
        ----------
        title : str
            The title.
        input_tensor_torch : torch.Tensor
            The input tensor.
        """
        input_tensor_numpy = input_tensor_torch.cpu().numpy()  # (batch, multi_step_input, values, variables)

        assert len(input_tensor_numpy.shape) == 4, input_tensor_numpy.shape
        assert input_tensor_numpy.shape[0] == 1, input_tensor_numpy.shape

        input_tensor_numpy = np.squeeze(input_tensor_numpy, axis=0)  # Drop the batch dimension
        input_tensor_numpy = np.swapaxes(input_tensor_numpy, -2, -1)  # (multi_step_input, variables, values)

        self._print_tensor(title, input_tensor_numpy, self._input_tensor_by_name, self._input_kinds)

    def _print_output_tensor(self, title: str, output_tensor_numpy: FloatArray) -> None:
        """Print the output tensor.

        Parameters
        ----------
        title : str
            The title.
        output_tensor_numpy : FloatArray
            The output tensor.
        """
        LOG.info(
            "%s",
            f"Output tensor shape={output_tensor_numpy.shape}, NaNs={np.isnan(output_tensor_numpy).sum() / output_tensor_numpy.size: .0%}",
        )

        if not self._output_tensor_by_name:
            for i in range(output_tensor_numpy.shape[1]):
                self._output_tensor_by_name.append(self.checkpoint.output_tensor_index_to_variable[i])
                if i in self.checkpoint.prognostic_output_mask:
                    self._output_kinds[self.checkpoint.output_tensor_index_to_variable[i]] = Kind(prognostic=True)
                else:
                    self._output_kinds[self.checkpoint.output_tensor_index_to_variable[i]] = Kind(diagnostic=True)

        # output_tensor_numpy = output_tensor_numpy.cpu().numpy()

        if len(output_tensor_numpy.shape) == 2:
            output_tensor_numpy = output_tensor_numpy[np.newaxis, ...]  # Add multi_step_input

        output_tensor_numpy = np.swapaxes(output_tensor_numpy, -2, -1)  # (multi_step_input, variables, values)

        self._print_tensor(title, output_tensor_numpy, self._output_tensor_by_name, self._output_kinds)

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
