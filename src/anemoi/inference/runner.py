# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
from functools import cached_property

import numpy as np
import torch
from anemoi.transform.grids.unstructured import UnstructuredGridFieldList
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from anemoi.utils.text import table
from anemoi.utils.timer import Timer  # , Timers

from .checkpoint import Checkpoint
from .context import Context
from .postprocess import Accumulator
from .postprocess import Noop
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


class Runner(Context):
    """_summary_"""

    _verbose = True

    def __init__(
        self,
        checkpoint,
        *,
        accumulations=True,
        device: str,
        precision: str = None,
        report_error=False,
        allow_nans=None,  # can be True of False
        verbosity=1,
    ):
        self._checkpoint = Checkpoint(checkpoint)

        self.device = device
        self.precision = precision
        self.report_error = report_error
        self.allow_nans = allow_nans

        # This could also be passed as an argument

        self.postprocess = Noop()

        if accumulations is True:
            # Get accumulations from the checkpoint
            accumulations = self.checkpoint.accumulations

        if accumulations:
            self.postprocess = Accumulator(accumulations)

        self.dynamic_forcings_sources = self.checkpoint.dynamic_forcings_sources(self)
        self._input_kinds = {}
        self._input_tensor_by_name = []

        self._output_kinds = {}
        self._output_tensor_by_name = []

        self.checkpoint.print_indices()

        LOG.info("Using %s runner", self.__class__.__name__)

    @property
    def checkpoint(self):
        return self._checkpoint

    def run(self, *, input_state, lead_time):

        input_state = self.validate_input_state(input_state)

        # timers = Timers()

        lead_time = to_timedelta(lead_time)

        input_tensor = self.prepare_input_tensor(input_state)

        try:
            yield from self.postprocess(self.forecast(lead_time, input_tensor, input_state))
        except (TypeError, ModuleNotFoundError):
            if self.report_error:
                self.checkpoint.report_error()
            raise

        # timers.report()

    def add_initial_forcings_to_input_state(self, input_state):
        latitudes = input_state["latitudes"]
        longitudes = input_state["longitudes"]
        date = input_state["date"]
        fields = input_state["fields"]

        dates = [date + h for h in self.checkpoint.lagged]

        # We allow user provided fields to be used as forcings
        variables = [v for v in self.checkpoint.model_computed_variables if v not in fields]

        LOG.info("Computing initial forcings %s", variables)
        # LOG.info("Computing initial forcings %s", self.checkpoint._metadata.input_computed_forcing_variables)

        forcings = self.compute_forcings(
            latitudes=latitudes,
            longitudes=longitudes,
            dates=dates,
            variables=variables,
        )

        typed_variables = self.checkpoint.typed_variables

        for name, forcing in zip(variables, forcings):
            fields[name] = forcing.to_numpy(dtype=np.float32, flatten=True)
            self._input_kinds[name] = Kind(computed=True, constant=typed_variables[name].is_constant_in_time)

    def prepare_input_tensor(self, input_state, dtype=np.float32):

        typed_variables = self.checkpoint.typed_variables

        for name in input_state["fields"]:
            self._input_kinds[name] = Kind(input=True, constant=typed_variables[name].is_constant_in_time)

        # Add initial forcings to input state if needed
        self.add_initial_forcings_to_input_state(input_state)

        input_fields = input_state["fields"]

        input_tensor_numpy = np.full(
            shape=(
                len(self.checkpoint.lagged),
                self.checkpoint.number_of_input_features,
                self.checkpoint.number_of_grid_points,
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
            raise ValueError(f"Missing variables in input fields: {[mapping.get(_,_) for _ in missing]}")

        return input_tensor_numpy

    @cached_property
    def autocast(self):
        autocast = self.precision

        if autocast is None:
            autocast = self.checkpoint.precision

        if autocast is None:
            LOG.warning("No autocast given, using float16")
            autocast = "16"

        return PRECISIONS.get(autocast, autocast)

    @cached_property
    def model(self):
        with Timer(f"Loading {self.checkpoint}"):
            return torch.load(self.checkpoint.path, map_location=self.device, weights_only=False).to(self.device)

    def forecast(self, lead_time, input_tensor_numpy, input_state):
        self.model.eval()

        torch.set_grad_enabled(False)

        # Create pytorch input tensor
        input_tensor_torch = torch.from_numpy(np.swapaxes(input_tensor_numpy, -2, -1)[np.newaxis, ...]).to(self.device)

        LOG.info("Using autocast %s", self.autocast)

        lead_time = to_timedelta(lead_time)
        steps = lead_time // self.checkpoint.frequency

        LOG.info("Lead time: %s, frequency: %s Forecasting %s steps", lead_time, self.checkpoint.frequency, steps)

        result = input_state.copy()  # We should not modify the input state
        result["fields"] = dict()

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
        self._print_input_tensor("First input tensor", input_tensor_torch)

        for s in range(steps):
            step = (s + 1) * self.checkpoint.frequency
            date = start + step
            LOG.info("Forecasting step %s (%s)", step, date)

            result["date"] = date

            # Predict next state of atmosphere
            with torch.autocast(device_type=self.device, dtype=self.autocast):
                y_pred = self.model.predict_step(input_tensor_torch)

            # Detach tensor and squeeze (should we detach here?)
            output = np.squeeze(y_pred.cpu().numpy())  # shape: (values, variables)

            # Update state
            for i in range(output.shape[1]):
                result["fields"][self.checkpoint.output_tensor_index_to_variable[i]] = output[:, i]

            if s == 0:
                self._print_output_tensor("Output tensor", output)

            yield result

            # Update  tensor for next iteration

            check[:] = reset

            self.copy_prognostic_fields_to_input_tensor(input_tensor_torch, y_pred, check)

            del y_pred  # Recover memory

            self.add_dynamic_forcings_to_input_tensor(input_tensor_torch, input_state, date, check)

            if not check.all():
                # Not all variables have been updated
                missing = []
                variable_to_input_tensor_index = self.checkpoint.variable_to_input_tensor_index
                mapping = {v: k for k, v in variable_to_input_tensor_index.items()}
                for i in range(check.shape[-1]):
                    if not check[i]:
                        missing.append(mapping[i])

                raise ValueError(f"Missing variables in input tensor: {sorted(missing)}")

            if s == 0:
                self._print_input_tensor("Next input tensor", input_tensor_torch)

    def copy_prognostic_fields_to_input_tensor(self, input_tensor_torch, y_pred, check):

        # input_tensor_torch is shape: (batch, lagged, variables, values)
        # batch is always 1

        prognostic_output_mask = self.checkpoint.prognostic_output_mask
        prognostic_input_mask = self.checkpoint.prognostic_input_mask

        # Copy prognostic fields to input tensor
        prognostic_fields = y_pred[..., prognostic_output_mask]  # Get new predicted values
        input_tensor_torch = input_tensor_torch.roll(-1, dims=1)  # Roll the tensor in the lagged dimension
        input_tensor_torch[:, -1, :, self.checkpoint.prognostic_input_mask] = (
            prognostic_fields  # Add new values to last 'lagged' row
        )

        assert not check[prognostic_input_mask].any()  # Make sure we are not overwriting some values
        check[prognostic_input_mask] = True

        for n in prognostic_input_mask:
            self._input_kinds[self._input_tensor_by_name[n]] = Kind(prognostic=True)

    def add_dynamic_forcings_to_input_tensor(self, input_tensor_torch, state, date, check):

        # input_tensor_torch is shape: (batch, lagged, variables, values)
        # batch is always 1

        for source in self.dynamic_forcings_sources:
            forcings = source.load_forcings(state, date)  # shape: (variables, values)

            forcings = np.swapaxes(forcings[np.newaxis, np.newaxis, ...], -2, -1)  # shape: (1, 1, values, variables)

            forcings = torch.from_numpy(forcings).to(self.device)  # Copy to device

            input_tensor_torch[:, -1, :, source.mask] = forcings  # Copy forcings to last 'lagged' row

            assert not check[source.mask].any()  # Make sure we are not overwriting some values
            check[source.mask] = True

            for n in source.mask:
                self._input_kinds[self._input_tensor_by_name[n]] = Kind(forcing=True, **source.kinds)

    def compute_forcings(self, *, latitudes, longitudes, dates, variables):
        import earthkit.data as ekd

        source = UnstructuredGridFieldList.from_values(latitudes=latitudes, longitudes=longitudes)

        ds = ekd.from_source("forcings", source, date=dates, param=variables)

        assert len(ds) == len(variables) * len(dates), (len(ds), len(variables), dates)

        return ds

    def validate_input_state(self, input_state):

        if not isinstance(input_state, dict):
            raise ValueError("Input state must be a dictionnary")

        EXPECT = dict(date=datetime.datetime, latitudes=np.ndarray, longitudes=np.ndarray, fields=dict)

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

        for latlon in ("latitudes", "longitudes"):
            if len(input_state[latlon].shape) != 1:
                raise ValueError(f"Input state entry `{latlon}` must be 1D, shape is {input_state[latlon].shape}")

        nlat = len(input_state["latitudes"])
        nlon = len(input_state["longitudes"])
        if nlat != nlon:
            raise ValueError(f"Size mismatch latitudes={nlat}, longitudes={nlon}")

        number_of_grid_points = nlat

        multi_step = len(self.checkpoint.lagged)

        expected_shape = (multi_step, number_of_grid_points)

        # Check field
        with_nans = []

        for name, field in list(fields.items()):

            # Allow for 1D fields if multi_step is 1
            if len(field.shape) == 1:
                field = fields[name] = field.reshape(1, field.shape[0])

            if field.shape != expected_shape:
                raise ValueError(f"Field `name` has the wrong shape. Expected {expected_shape}, got {field.shape}")

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

        # Needed for some output object, such as GribOutput, to compute `step`

        input_state["reference_date"] = input_state["date"]

        return input_state

    def _print_tensor(self, title, tensor_numpy, tensor_by_name, kinds):

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

    def _print_input_tensor(self, title, input_tensor_torch):

        input_tensor_numpy = input_tensor_torch.cpu().numpy()  # (batch, lagged, values, variables)

        assert len(input_tensor_numpy.shape) == 4, input_tensor_numpy.shape
        assert input_tensor_numpy.shape[0] == 1, input_tensor_numpy.shape

        print(input_tensor_numpy.shape)
        input_tensor_numpy = np.squeeze(input_tensor_numpy, axis=0)  # Drop the batch dimension

        print(input_tensor_numpy.shape)
        input_tensor_numpy = np.swapaxes(input_tensor_numpy, -2, -1)  # (lagged, variables, values)

        self._print_tensor(title, input_tensor_numpy, self._input_tensor_by_name, self._input_kinds)

    def _print_output_tensor(self, title, output_tensor_numpy):

        if not self._output_tensor_by_name:
            for i in range(output_tensor_numpy.shape[1]):
                self._output_tensor_by_name.append(self.checkpoint.output_tensor_index_to_variable[i])
                if i in self.checkpoint.prognostic_output_mask:
                    self._output_kinds[self.checkpoint.output_tensor_index_to_variable[i]] = Kind(prognostic=True)
                else:
                    self._output_kinds[self.checkpoint.output_tensor_index_to_variable[i]] = Kind(diagnostic=True)

        if isinstance(output_tensor_numpy, torch.Tensor):
            output_tensor_numpy = output_tensor_numpy.cpu().numpy()

        self._print_tensor(title, output_tensor_numpy, self._output_tensor_by_name, self._output_kinds)
