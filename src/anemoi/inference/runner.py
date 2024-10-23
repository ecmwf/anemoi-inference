# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import warnings
from collections import defaultdict
from functools import cached_property

import numpy as np
import torch
from anemoi.transform.grids.unstructured import UnstructuredGridFieldList
from anemoi.utils.dates import frequency_to_timedelta
from anemoi.utils.humanize import plural
from anemoi.utils.timer import Timer
from earthkit.data.indexing.fieldlist import FieldArray

from .checkpoint import Checkpoint
from .precisions import PRECISIONS

LOG = logging.getLogger(__name__)


class Runner:
    """_summary_"""

    _verbose = True

    def __init__(self, checkpoint, *, device: str, precision: str = None, verbose: bool = True):
        self.checkpoint = Checkpoint(checkpoint, verbose=verbose)
        self._verbose = verbose
        self.device = device
        self.precision = precision

    def run(self, *, input_state, lead_time):

        input_state = self.prepare_input_state(input_state)
        input_tensor = self.prepare_input_tensor(input_state)
        yield from self.forecast(lead_time, input_tensor, input_state)

    def prepare_input_state(self, input_fields, start_datetime=None, dtype=np.float32, flatten=True):

        input_state = dict()

        if start_datetime is None:
            start_datetime = input_fields.order_by(valid_datetime="ascending")[-1].datetime()["valid_time"]
            LOG.info("start_datetime not provided, using %s as start_datetime", start_datetime.isoformat())

        dates = [start_datetime + h for h in self.checkpoint.lagged]
        date_to_index = {d.isoformat(): i for i, d in enumerate(dates)}

        input_state["dates"] = dates
        fields = input_state["fields"] = dict()

        input_fields = self.filter_and_sort(input_fields, dates)

        check = defaultdict(set)

        first = True
        for field in input_fields:

            if first:
                first = False
                input_state["latitudes"], input_state["longitudes"] = field.grid_points()

            name, valid_datetime = field.metadata("name"), field.metadata("valid_datetime")
            if name not in fields:
                fields[name] = np.full(
                    shape=(len(dates), self.checkpoint.number_of_grid_points),
                    fill_value=np.nan,
                    dtype=dtype,
                )

            date_idx = date_to_index[valid_datetime]
            fields[name][date_idx] = field.to_numpy(dtype=dtype, flatten=flatten)

            if date_idx in check[name]:
                LOG.error("Duplicate dates for %s: %s", name, date_idx)
                LOG.error("Expected %s", list(date_to_index.keys()))
                LOG.error("Got %s", list(check[name]))
                raise ValueError(f"Duplicate dates for {name}")

            check[name].add(date_idx)

        for name, idx in check.items():
            if len(idx) != len(dates):
                LOG.error("Missing dates for %s: %s", name, idx)
                LOG.error("Expected %s", list(date_to_index.keys()))
                LOG.error("Got %s", list(idx))
                raise ValueError(f"Missing dates for {name}")

        return input_state

    def add_initial_forcings_to_input_state(self, input_state):
        latitudes = input_state["latitudes"]
        longitudes = input_state["longitudes"]
        dates = input_state["dates"]
        fields = input_state["fields"]

        # We allow user provided fields to be used as forcings
        variables = [v for v in self.checkpoint.model_computed_variables if v not in fields]

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
        dates = input_state["dates"]

        input_tensor_numpy = np.full(
            shape=(
                len(dates),
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
                raise ValueError(f"Duplicate variable {var}/i={i}")

            input_tensor_numpy[:, i] = field
            check.add(i)

        missing = set(range(self.checkpoint.number_of_input_features)) - check

        if missing:
            index_to_variable = self.checkpoint.model_index_to_variable
            LOG.error("Missing variables %s", [index_to_variable[i] for i in missing])
            raise ValueError(f"Missing variables {missing}")

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
            try:
                return torch.load(self.checkpoint.path, map_location=self.device, weights_only=False).to(self.device)
            except Exception:
                self.checkpoint.report_loading_error()
                raise

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
        print(result.keys())

        start_datetime = (
            max(input_state["dates"]) if isinstance(input_state["dates"], (list, tuple)) else input_state["dates"]
        )

        for i in range(steps):
            step = (i + 1) * self.checkpoint.frequency
            date = start_datetime + step
            LOG.info("Forecasting step %s (%s)", step, date)

            result["dates"] = [date]

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

    def filter_and_sort(self, data, dates):
        typed_variables = self.checkpoint.typed_variables
        diagnostic_variables = self.checkpoint.diagnostic_variables

        def _name(field, key, original_metadata):
            warnings.warn("TEMPORARY CODE: Use the remapping in the metadata")
            param, levelist, levtype = (
                original_metadata.get("param"),
                original_metadata.get("levelist"),
                original_metadata.get("levtype"),
            )

            # Bug in eccodes that returns levelist for single level fields in GRIB2
            if levtype in ("sfc", "o2d"):
                levelist = None

            if levelist is None:
                return param

            return f"{param}_{levelist}"

        data = FieldArray([f.copy(name=_name) for f in data])

        variable_from_input = [
            v.name for v in typed_variables.values() if v.is_from_input and v.name not in diagnostic_variables
        ]

        valid_datetime = [_.isoformat() for _ in dates]
        LOG.info("Selecting fields %s %s", len(data), valid_datetime)

        data = data.sel(name=variable_from_input, valid_datetime=valid_datetime).order_by("name", "valid_datetime")

        expected = len(variable_from_input) * len(dates)

        if len(data) != expected:
            nvars = plural(len(variable_from_input), "variable")
            ndates = plural(len(dates), "date")
            nfields = plural(expected, "field")
            msg = f"Expected ({nvars}) x ({ndates}) = {nfields}, got {len(data)}"
            LOG.error("%s", msg)
            # TODO: print a report
            raise ValueError(msg)

        assert len(data) == len(variable_from_input) * len(dates)

        return data

    def compute_forcings(self, *, latitudes, longitudes, dates, variables):
        import earthkit.data as ekd

        source = UnstructuredGridFieldList.from_values(latitudes=latitudes, longitudes=longitudes)

        ds = ekd.from_source("forcings", source, date=dates, param=variables)

        assert len(ds) == len(variables) * len(dates), (len(ds), len(variables), dates)

        return ds
