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
from functools import cached_property

import numpy as np
import torch
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from anemoi.utils.text import table
from anemoi.utils.timer import Timer  # , Timers

from .context import Context
from .ds_checkpoint import Checkpoint_0, Checkpoint_1
from .postprocess import Accumulator, Noop
from .precisions import PRECISIONS

LOG = logging.getLogger(__name__)


class Kind:
    """Used for debugging purposes"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __repr__(self):
        result = []

        for k, v in self.kwargs.items():
            if v:
                result.append(k)

        if not result:
            return "?"

        return ", ".join(result)


class DownscalingRunner(Context):
    """A runner is responsible for running a model."""

    def __init__(
        self,
        checkpoint,
        *,
        accumulations=True,
        device: str,
        precision: str = None,
        report_error=False,
        allow_nans=None,  # can be True of False
        use_grib_paramid=False,
        verbosity=0,
    ):
        self._checkpoint_0 = Checkpoint_0(checkpoint)
        self._checkpoint_1 = Checkpoint_1(checkpoint)

        self.device = device
        self.precision = precision
        self.report_error = report_error

        # Override the default values set in `Context`
        self.verbosity = verbosity
        self.allow_nans = allow_nans
        self.use_grib_paramid = use_grib_paramid

        # This could also be passed as an argument

        self.postprocess = Noop()

        if accumulations is True:
            accumulations = self.checkpoint_0.accumulations

        if accumulations:   
            self.postprocess = Accumulator(accumulations)

        self._input_0_kinds = {}
        self._input_1_kinds = {}
        self._input_0_tensor_by_name = []
        self._input_1_tensor_by_name = []

        self._forcing_kinds = {}
        self._forcing_tensor_by_name = []

        self._output_kinds = {}
        self._output_tensor_by_name = []

        if self.verbosity > 2:
            self.checkpoint_0.print_indices()

        LOG.info("Using %s runner", self.__class__.__name__)

        if self.verbosity > 1:
            self.checkpoint_0.print_variable_categories()

    @property
    def checkpoint(self):
        return self._checkpoint_0
    
    @property
    def checkpoint_0(self):
        return self._checkpoint_0

    @property
    def checkpoint_1(self):
        return self._checkpoint_1


    def run(self, *, input_0_state, input_1_state):

        self.constant_forcings_input_0 = self.checkpoint_0.constant_forcings_inputs(
            self, input_0_state
        )
        self.dynamic_forcings_input_0 = self.checkpoint_0.dynamic_forcings_inputs(
            self, input_0_state
        )
        self.boundary_forcings_input_0 = self.checkpoint_0.boundary_forcings_inputs(
            self, input_0_state
        )

        self.constant_forcings_input_1 = self.checkpoint_1.constant_forcings_inputs(
            self, input_1_state
        )
        self.dynamic_forcings_input_1 = self.checkpoint_1.dynamic_forcings_inputs(
            self, input_1_state
        )
        self.boundary_forcings_input_1 = self.checkpoint_1.boundary_forcings_inputs(
            self, input_1_state
        )

        # timers = Timers()
        input_0_tensor = self.prepare_input_tensor(input_0_state, 0)
        input_1_tensor = self.prepare_input_tensor(input_1_state, 1)

        try:
            yield from self.postprocess(
                self.downscale(
                    input_0_tensor, input_1_tensor, input_0_state, input_1_state
                )
            )
        except (TypeError, ModuleNotFoundError):
            if self.report_error:
                self.checkpoint_0.report_error()
            raise

        # timers.report()

    def add_initial_forcings_to_input_state(self, input_state, idx_input):

        constant_forcings = getattr(self, f"constant_forcings_input_{idx_input}")
        dynamic_forcings = getattr(self, f"dynamic_forcings_input_{idx_input}")

        # Should that be alreay a list of dates
        date = input_state["date"]
        fields = input_state["fields"]

        dates = [date + h for h in self.checkpoint_0.lagged]

        # For output object. Should be moved elsewhere
        self.reference_date = dates[-1]
        self.initial_dates = dates

        # TODO: Check for user provided forcings

        for source in constant_forcings:
            LOG.info(
                "Constant forcings input: %s %s (%s)", source, source.variables, dates
            )
            arrays = source.load_forcings(input_state, dates)
            for name, forcing in zip(source.variables, arrays):
                assert isinstance(forcing, np.ndarray), (name, forcing)
                fields[name] = forcing
                if idx_input == 0:
                    self._input_0_kinds[name] = Kind(
                        forcing=True, constant=True, **source.kinds
                    )
                elif idx_input == 1:
                    self._input_1_kinds[name] = Kind(
                        forcing=True, constant=True, **source.kinds
                    )

        for source in dynamic_forcings:
            LOG.info(
                "Dynamic forcings input: %s %s (%s)", source, source.variables, dates
            )
            arrays = source.load_forcings(input_state, dates)
            for name, forcing in zip(source.variables, arrays):
                assert isinstance(forcing, np.ndarray), (name, forcing)
                fields[name] = forcing
                if idx_input == 0:
                    self._input_0_kinds[name] = Kind(
                        forcing=True, constant=True, **source.kinds
                    )
                elif idx_input == 1:
                    self._input_1_kinds[name] = Kind(
                        forcing=True, constant=True, **source.kinds
                    )

    def prepare_input_tensor(self, input_state, idx_input, dtype=np.float32):

        for name in input_state["fields"]:
            if idx_input == 0:
                self._input_0_kinds[name] = Kind(
                    input=True, constant=self.checkpoint_0.typed_variables[name].is_constant_in_time
                )
            elif idx_input == 1:
                self._input_1_kinds[name] = Kind(
                    input=True, constant=self.checkpoint_1.typed_variables[name].is_constant_in_time
                )
            else:
                raise ValueError(f"Invalid idx_input {idx_input}")

            checkpoint = getattr(self, f"checkpoint_{idx_input}")
            number_of_grid_points = checkpoint.number_of_input_grid_points
            number_of_features = checkpoint.number_of_input_features
            variable_to_input_tensor_index = (
                checkpoint.variable_to_input_tensor_index
            )


        # Add initial forcings to input state if needed
        self.add_initial_forcings_to_input_state(input_state, idx_input)

        input_state = self.validate_input_state(input_state, idx_input)

        input_fields = input_state["fields"]

        input_tensor_numpy = np.full(
            shape=(
                1,
                number_of_features,
                number_of_grid_points,
            ),
            fill_value=np.nan,
            dtype=dtype,
        )

        self._input_tensor_by_name = [None] * number_of_features

        LOG.info("Preparing input tensor with shape %s", input_tensor_numpy.shape)

        check = set()
        for var, field in input_fields.items():
            i = variable_to_input_tensor_index[var]
            if i in check:
                raise ValueError(f"Duplicate variable {var}/{i} in input fields")
            input_tensor_numpy[:, i] = field
            check.add(i)

            self._input_tensor_by_name[i] = var

        if len(check) != number_of_features:
            missing = set(range(number_of_features)) - check
            mapping = {v: k for k, v in variable_to_input_tensor_index.items()}
            raise ValueError(
                f"Missing variables in input fields: {[mapping.get(_,_) for _ in missing]}"
            )

        return input_tensor_numpy

    @cached_property
    def autocast(self):
        autocast = self.precision

        if autocast is None:
            autocast = self.checkpoint_0.precision

        if autocast is None:
            LOG.warning("No autocast given, using float16")
            autocast = "16"

        return PRECISIONS.get(autocast, autocast)

    @cached_property
    def model(self):
        with Timer(f"Loading {self.checkpoint_0}"):
            return torch.load(
                self.checkpoint_0.path, map_location=self.device, weights_only=False
            ).to(self.device)

    def downscale(
        self,
        input_tensor_numpy_0,
        input_tensor_numpy_1,
        input_state_0,
        input_state_1,
    ):
        self.model.eval()

        torch.set_grad_enabled(False)

        # Create pytorch input tensor
        input_tensor_torch_0 = torch.from_numpy(
            np.swapaxes(input_tensor_numpy_0, -2, -1)[np.newaxis, ...]
        ).to(self.device)
        input_tensor_torch_1 = torch.from_numpy(
            np.swapaxes(input_tensor_numpy_1, -2, -1)[np.newaxis, ...]
        ).to(self.device)

        LOG.info("Using autocast %s", self.autocast)

        steps = 1

        result = input_state_0.copy()  # We should not modify the input state
        result["fields"] = dict()

        start = input_state_0["date"]

        # The variable `check` is used to keep track of which variables have been updated
        # In the input tensor. `reset` is used to reset `check` to False except
        # when the values are of the constant in time variables

        reset = np.full((input_tensor_torch_0.shape[-1],), False)
        variable_to_input_tensor_index = (
            self.checkpoint_0.variable_to_input_tensor_index
        )
        typed_variables = self.checkpoint_0.typed_variables
        for variable, i in variable_to_input_tensor_index.items():
            if typed_variables[variable].is_constant_in_time:
                reset[i] = True

        check = reset.copy()

        if self.verbosity > 0:
            self._print_input_tensor("First input tensor", input_tensor_torch_0, 0)

        for s in range(steps):
            step = (s + 1) * self.checkpoint_0.timestep
            date = start + step
            LOG.info("Forecasting step %s (%s)", step, date)

            result["date"] = date

            # Predict next state of atmosphere
            with torch.autocast(device_type=self.device, dtype=self.autocast):
                y_pred = self.model.predict_step(
                    [input_tensor_torch_0, input_tensor_torch_1]
                )

            # Detach tensor and squeeze (should we detach here?)
            output = np.squeeze(y_pred.cpu().numpy())  # shape: (values, variables)

            # Update state
            for i in range(output.shape[1]):
                result["fields"][self.checkpoint_0.output_tensor_index_to_variable[i]] = (
                    output[:, i]
                )

            if (s == 0 and self.verbosity > 0) or self.verbosity > 1:
                self._print_output_tensor("Output tensor", output)

            yield result

            """
            # Update  tensor for next iteration

            check[:] = reset

            input_tensor_torch = self.copy_prognostic_fields_to_input_tensor(
                input_tensor_torch, y_pred, check
            )

            del y_pred  # Recover memory

            input_tensor_torch = self.add_dynamic_forcings_to_input_tensor(
                input_tensor_torch, input_state, date, check
            )
            input_tensor_torch = self.add_boundary_forcings_to_input_tensor(
                input_tensor_torch, input_state, date, check
            )

            if not check.all():
                # Not all variables have been updated
                missing = []
                variable_to_input_tensor_index = (
                    self.checkpoint.variable_to_input_tensor_index
                )
                mapping = {v: k for k, v in variable_to_input_tensor_index.items()}
                for i in range(check.shape[-1]):
                    if not check[i]:
                        missing.append(mapping[i])

                raise ValueError(
                    f"Missing variables in input tensor: {sorted(missing)}"
                )

            if (s == 0 and self.verbosity > 0) or self.verbosity > 1:
                self._print_input_tensor("Next input tensor", input_tensor_torch)
            """

    """
    def copy_prognostic_fields_to_input_tensor(self, input_tensor_torch, y_pred, check):

        # input_tensor_torch is shape: (batch, multi_step_input, values, variables)
        # batch is always 1

        prognostic_output_mask = self.checkpoint.prognostic_output_mask
        prognostic_input_mask = self.checkpoint.prognostic_input_mask

        # Copy prognostic fields to input tensor
        prognostic_fields = y_pred[
            ..., prognostic_output_mask
        ]  # Get new predicted values
        input_tensor_torch = input_tensor_torch.roll(
            -1, dims=1
        )  # Roll the tensor in the multi_step_input dimension
        input_tensor_torch[:, -1, :, self.checkpoint.prognostic_input_mask] = (
            prognostic_fields  # Add new values to last 'multi_step_input' row
        )

        assert not check[
            prognostic_input_mask
        ].any()  # Make sure we are not overwriting some values
        check[prognostic_input_mask] = True

        for n in prognostic_input_mask:
            self._input_kinds[self._input_tensor_by_name[n]] = Kind(prognostic=True)

        return input_tensor_torch
    """

    """
    def add_dynamic_forcings_to_input_tensor(
        self, input_tensor_torch, state, date, check
    ):

        # input_tensor_torch is shape: (batch, multi_step_input, values, variables)
        # batch is always 1

        for source in self.dynamic_forcings_inputs:
            forcings = source.load_forcings(
                state, [date]
            )  # shape: (variables, dates, values)

            forcings = np.squeeze(forcings, axis=1)  # Drop the dates dimension

            forcings = np.swapaxes(
                forcings[np.newaxis, np.newaxis, ...], -2, -1
            )  # shape: (1, 1, values, variables)

            forcings = torch.from_numpy(forcings).to(self.device)  # Copy to device

            input_tensor_torch[:, -1, :, source.mask] = (
                forcings  # Copy forcings to last 'multi_step_input' row
            )

            assert not check[
                source.mask
            ].any()  # Make sure we are not overwriting some values
            check[source.mask] = True

            for n in source.mask:
                self._input_kinds[self._input_tensor_by_name[n]] = Kind(
                    forcing=True, **source.kinds
                )

        return input_tensor_torch
    """
    """
    def add_boundary_forcings_to_input_tensor(
        self, input_tensor_torch, state, date, check
    ):

        # input_tensor_torch is shape: (batch, multi_step_input, values, variables)
        # batch is always 1
        sources = self.boundary_forcings_inputs
        for source in sources:
            forcings = source.load_forcings(
                state, [date]
            )  # shape: (variables, dates, values)

            forcings = np.squeeze(forcings, axis=1)  # Drop the dates dimension

            forcings = np.swapaxes(
                forcings[np.newaxis, np.newaxis, ...], -2, -1
            )  # shape: (1, 1, values, variables)
            forcings = torch.from_numpy(forcings).to(self.device)  # Copy to device
            total_mask = np.ix_([0], [-1], source.spatial_mask, source.variables_mask)
            input_tensor_torch[total_mask] = (
                forcings  # Copy forcings to last 'multi_step_input' row
            )

        # TO DO: add some consistency checks as above
        return input_tensor_torch
    """

    def validate_input_state(self, input_state, idx_input):

        if not isinstance(input_state, dict):
            raise ValueError("Input state must be a dictionnary")

        EXPECT = dict(
            date=datetime.datetime,
            latitudes=np.ndarray,
            longitudes=np.ndarray,
            fields=dict,
        )

        for key, klass in EXPECT.items():
            if key not in input_state:
                raise ValueError(f"Input state must contain a `{key}` enytry")

            if not isinstance(input_state[key], klass):
                raise ValueError(
                    f"Input state entry `{key}` is type {type(input_state[key])}, expected {klass} instead"
                )

        # Detach from the user's input so we can modify it
        input_state = input_state.copy()
        fields = input_state["fields"] = input_state["fields"].copy()
        if idx_input == 0:
            number_of_grid_points = self.checkpoint_0.number_of_input_grid_points
        elif idx_input == 1:
            number_of_grid_points = self.checkpoint_1.number_of_input_grid_points
        

        for latlon in ("latitudes", "longitudes"):
            if len(input_state[latlon].shape) != 1:
                raise ValueError(
                    f"Input state entry `{latlon}` must be 1D, shape is {input_state[latlon].shape}"
                )

        nlat = len(input_state["latitudes"])
        nlon = len(input_state["longitudes"])
        if nlat != nlon:
            raise ValueError(f"Size mismatch latitudes={nlat}, longitudes={nlon}")

        if nlat != number_of_grid_points:
            raise ValueError(
                f"Size mismatch latitudes={nlat}, number_of_grid_points={number_of_grid_points}"
            )

        multi_step = 1

        expected_shape = (multi_step, number_of_grid_points)

        LOG.info("Expected shape for each input fields: %s", expected_shape)

        # Check field
        with_nans = []

        for name, field in list(fields.items()):

            # Allow for 1D fields if multi_step is 1
            if len(field.shape) == 1:
                field = fields[name] = field.reshape(1, field.shape[0])

            if field.shape != expected_shape:
                raise ValueError(
                    f"Field `{name}` has the wrong shape. Expected {expected_shape}, got {field.shape}"
                )

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

    def _print_tensor(self, title, tensor_numpy, tensor_by_name, kinds):

        assert len(tensor_numpy.shape) == 3, tensor_numpy.shape
        assert tensor_numpy.shape[0] in (
            1,
            1,
        ), tensor_numpy.shape
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

            t.append(
                (k, v, np.nanmin(data), np.nanmax(data), nans, kinds.get(v, Kind()))
            )

        LOG.info("")
        LOG.info(
            "%s:\n\n%s\n",
            title,
            table(
                t,
                header=["Index", "Variable", "Min", "Max", "NaNs", "Kind"],
                align="><<<|<",
            ),
        )
        LOG.info("")

    def _print_input_tensor(self, title, input_tensor_torch, idx_input):

        input_tensor_numpy = (
            input_tensor_torch.cpu().numpy()
        )  # (batch, multi_step_input, values, variables)

        assert len(input_tensor_numpy.shape) == 4, input_tensor_numpy.shape
        assert input_tensor_numpy.shape[0] == 1, input_tensor_numpy.shape

        input_tensor_numpy = np.squeeze(
            input_tensor_numpy, axis=0
        )  # Drop the batch dimension
        input_tensor_numpy = np.swapaxes(
            input_tensor_numpy, -2, -1
        )  # (multi_step_input, variables, values)

        self._print_tensor(
            title,
            input_tensor_numpy,
            self._input_tensor_by_name,
            getattr(self, f"_input_{idx_input}_kinds"),
        )

    def _print_output_tensor(self, title, output_tensor_numpy):

        LOG.info(
            "%s",
            f"Output tensor shape={output_tensor_numpy.shape}, NaNs={np.isnan(output_tensor_numpy).sum()/ output_tensor_numpy.size: .0%}",
        )

        if not self._output_tensor_by_name:
            for i in range(output_tensor_numpy.shape[1]):
                self._output_tensor_by_name.append(
                    self.checkpoint_0.output_tensor_index_to_variable[i]
                )
                if i in self.checkpoint_0.prognostic_output_mask:
                    self._output_kinds[
                        self.checkpoint_0.output_tensor_index_to_variable[i]
                    ] = Kind(prognostic=True)
                else:
                    self._output_kinds[
                        self.checkpoint_0.output_tensor_index_to_variable[i]
                    ] = Kind(diagnostic=True)

        # output_tensor_numpy = output_tensor_numpy.cpu().numpy()

        if len(output_tensor_numpy.shape) == 2:
            output_tensor_numpy = output_tensor_numpy[
                np.newaxis, ...
            ]  # Add multi_step_input

        output_tensor_numpy = np.swapaxes(
            output_tensor_numpy, -2, -1
        )  # (multi_step_input, variables, values)

        self._print_tensor(
            title, output_tensor_numpy, self._output_tensor_by_name, self._output_kinds
        )
