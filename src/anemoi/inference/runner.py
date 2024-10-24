# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from functools import cached_property

import numpy as np
import torch
from anemoi.transform.grids.unstructured import UnstructuredGridFieldList
from anemoi.utils.dates import frequency_to_timedelta
from anemoi.utils.timer import Timer

from .checkpoint import Checkpoint
from .forcings import ComputedForcings
from .precisions import PRECISIONS

LOG = logging.getLogger(__name__)


class Noop:

    def __call__(self, source):
        yield from source


class Accumulator:
    """Accumulate fields from zero and return the accumulated fields"""

    def __init__(self, accumulations):
        self.accumulations = accumulations
        LOG.info("Accumulating fields %s", self.accumulations)

        self.accumulators = {}

    def __call__(self, source):
        for state in source:
            for accumulation in self.accumulations:
                if accumulation in state["fields"]:
                    self.accumulators[accumulation] = np.zeros_like(state["fields"][accumulation])
                self.accumulators[accumulation] += np.maximum(0, state["fields"][accumulation])
                state["fields"][accumulation] = self.accumulators[accumulation]

            yield state


class Runner:
    """_summary_"""

    _verbose = True

    def __init__(self, checkpoint, *, accumulations=True, device: str, precision: str = None, verbose: bool = True):
        self.checkpoint = Checkpoint(checkpoint, verbose=verbose)
        self._verbose = verbose
        self.device = device
        self.precision = precision

        # This could also be passed as an argument

        self.postprocess = Noop()

        if accumulations is True:
            # Get accumulations from the checkpoint
            accumulations = self.checkpoint.accumulations

        if accumulations:
            self.postprocess = Accumulator(accumulations)

        self.dynamic_forcings = []

        forcing_mask, forcing_variables = self.checkpoint.computed_time_dependent_forcings
        if len(forcing_mask) > 0:
            self.dynamic_forcings.append(ComputedForcings(self, forcing_variables))

    def run(self, *, input_state, lead_time):
        lead_time = frequency_to_timedelta(lead_time)

        input_tensor = self.prepare_input_tensor(input_state)

        try:
            yield from self.postprocess(self.forecast(lead_time, input_tensor, input_state))
        except (TypeError, ModuleNotFoundError):
            self.checkpoint.report_error()
            raise

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

        for name, forcing in zip(variables, forcings):
            fields[name] = forcing.to_numpy(dtype=np.float32, flatten=True)

    def prepare_input_tensor(self, input_state, dtype=np.float32):

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

        LOG.info("Preparing input tensor with shape %s", input_tensor_numpy.shape)

        variable_to_input_tensor_index = self.checkpoint.variable_to_input_tensor_index

        check = set()
        for var, field in input_fields.items():
            i = variable_to_input_tensor_index[var]
            if i in check:
                raise ValueError(f"Duplicate variable {var}/{i} in input fields")
            input_tensor_numpy[:, i] = field
            check.add(i)

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

        input_tensor_torch = torch.from_numpy(
            np.swapaxes(
                input_tensor_numpy,
                -2,
                -1,
            )[np.newaxis, ...]
        ).to(self.device)

        LOG.info("Using autocast %s", self.autocast)

        lead_time = frequency_to_timedelta(lead_time)
        steps = lead_time // self.checkpoint.frequency

        LOG.info("Lead time: %s, frequency: %s Forecasting %s steps", lead_time, self.checkpoint.frequency, steps)

        result = input_state.copy()
        result["fields"] = dict()

        start = input_state["date"]

        for i in range(steps):
            step = (i + 1) * self.checkpoint.frequency
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

            yield result

            # Update  tensor for next iteration

            # Copy prognostic fields to input tensor

            prognostic_fields = y_pred[..., self.checkpoint.prognostic_output_mask]
            input_tensor_torch = input_tensor_torch.roll(-1, dims=1)
            input_tensor_torch[:, -1, :, self.checkpoint.prognostic_input_mask] = prognostic_fields

            # Compute new forcings if needed

            forcing_mask, forcing_variables = self.checkpoint.computed_time_dependent_forcings

            if len(forcing_mask) > 0:

                forcing = self.compute_forcings(
                    latitudes=input_state["latitudes"],
                    longitudes=input_state["longitudes"],
                    variables=forcing_variables,
                    dates=[date],
                )

                forcing = forcing.to_numpy(dtype=np.float32, flatten=True)

                forcing = np.swapaxes(forcing[np.newaxis, np.newaxis, ...], -2, -1)
                forcing = torch.from_numpy(forcing).to(self.device)
                input_tensor_torch[:, -1, :, forcing_mask] = forcing

    def compute_forcings(self, *, latitudes, longitudes, dates, variables):
        import earthkit.data as ekd

        source = UnstructuredGridFieldList.from_values(latitudes=latitudes, longitudes=longitudes)

        ds = ekd.from_source("forcings", source, date=dates, param=variables)

        assert len(ds) == len(variables) * len(dates), (len(ds), len(variables), dates)

        return ds
