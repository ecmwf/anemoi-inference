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

LOG = logging.getLogger(__name__)


AUTOCAST = {
    "16-mixed": torch.float16,
    "16": torch.float16,
    "32": torch.float32,
    "b16-mixed": torch.bfloat16,
    "b16": torch.bfloat16,
    "bf16-mixed": torch.bfloat16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
    "float16": torch.float16,
    "float32": torch.float32,
}


def forcing_and_constants(*, latitudes, longitudes, date, param):
    import earthkit.data as ekd

    source = UnstructuredGridFieldList.from_values(latitudes=latitudes, longitudes=longitudes)

    ds = ekd.from_source(
        "forcings",
        source,
        date=date,
        param=param,
    )

    assert len(ds) == len(param) * len(date), (len(ds), len(param), date)

    return ds


def ignore(*args, **kwargs):
    pass


def _fix_eccodes_bug_for_levtype_sfc_and_grib2(input_fields):
    from earthkit.data.indexing.fieldlist import FieldArray

    class BugFix:
        def __init__(self, field):
            self.field = field

        def __getattr__(self, name):
            return getattr(self.field, name)

        def metadata(self, *args, remapping=None, patches=None, **kwargs):
            if remapping is not None or patches is not None:
                from earthkit.data.core.order import build_remapping

                remapping = build_remapping(remapping, patches)
                return remapping(self.metadata)(*args, **kwargs)

            if len(args) > 0 and args[0] == "levelist":
                if "default" in kwargs:
                    return kwargs["default"]
                raise KeyError("levelist")
            return self.field.metadata(*args, **kwargs)

        def __repr__(self) -> str:
            return repr(self.field)

    fixed = []
    for field in input_fields:
        if field.metadata("levtype") in ("sfc", "o2d") and field.metadata("edition") == 2:
            if field.metadata("levelist", default=None) is not None:
                # LOGGER.warning("Fixing eccodes bug for levtype=sfc and grib2 %s", field)
                fixed.append(BugFix(field))
        else:
            fixed.append(field)

    return FieldArray(fixed)


class Runner:
    """_summary_"""

    _verbose = True

    def __init__(self, checkpoint, verbose: bool = True):
        self.checkpoint = Checkpoint(checkpoint)
        self._verbose = verbose

    def run(
        self,
        *,
        input_state,
        lead_time,
        device,
        start_datetime=None,
        output_callback=ignore,
        autocast=None,
        progress_callback=ignore,
        grid_field_list=None,
    ) -> None:

        if not isinstance(input_state, dict):
            input_state = self.prepare_input_state(input_state, start_datetime)

        input_tensor = self.prepare_input_tensor(input_state)
        print(input_tensor.shape)

        self.forecast(
            lead_time, device, input_tensor, input_state, autocast=autocast, progress_callback=progress_callback
        )

    def prepare_input_state(self, input_fields, start_datetime, dtype=np.float32, flatten=True):
        """Convert an earthkit FieldArray to a dictionary of numpy arrays."""

        checkpoint_metadata = self.checkpoint.metadata

        input_state = dict()

        if start_datetime is None:
            start_datetime = input_fields.order_by(valid_datetime="ascending")[-1].datetime()["valid_time"]
            LOG.info("start_datetime not provided, using %s as start_datetime", start_datetime.isoformat())

        dates = [start_datetime + h for h in self.lagged]
        date_to_index = {d.isoformat(): i for i, d in enumerate(dates)}

        input_state["dates"] = dates
        fields = input_state["fields"] = dict()

        input_fields = _fix_eccodes_bug_for_levtype_sfc_and_grib2(input_fields)
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
                    shape=(len(dates), checkpoint_metadata.number_of_grid_points),
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

    def prepare_input_tensor(self, input_state, dtype=np.float32):
        checkpoint_metadata = self.checkpoint.metadata

        input_fields = input_state["fields"]
        dates = input_state["dates"]
        latitudes = input_state["latitudes"]
        longitudes = input_state["longitudes"]

        input_tensor_numpy = np.full(
            shape=(
                len(dates),
                checkpoint_metadata.number_of_input_features,
                checkpoint_metadata.number_of_grid_points,
            ),
            fill_value=np.nan,
            dtype=dtype,
        )

        LOG.info("Preparing input tensor with shape %s", input_tensor_numpy.shape)

        variable_to_input_tensor_index = checkpoint_metadata.variable_to_input_tensor_index

        check = set()

        for var, field in input_fields.items():
            i = variable_to_input_tensor_index[var]

            if i in check:
                raise ValueError(f"Duplicate variable {var}/i={i}")

            check.add(i)

            input_tensor_numpy[:, i] = field

        # Add computed forcing variables

        # TODO: extend the state instead
        model_computed_variables = checkpoint_metadata.model_computed_variables
        if model_computed_variables:
            fields = forcing_and_constants(
                latitudes=latitudes, longitudes=longitudes, date=dates, param=model_computed_variables
            )
            for var, field in zip(model_computed_variables, fields):
                i = variable_to_input_tensor_index[var]
                if i in check:
                    raise ValueError(f"Duplicate variable {var}/i={i}")
                check.add(i)
                input_tensor_numpy[:, i] = field.to_numpy(dtype=dtype, flatten=True)

        missing = set(range(checkpoint_metadata.number_of_input_features)) - check

        if missing:
            index_to_variable = checkpoint_metadata.model_index_to_variable
            LOG.error("Missing variables %s", [index_to_variable[i] for i in missing])

            assert not missing, missing

        return input_tensor_numpy

    def forecast(self, lead_time, device, input_tensor_numpy, input_state, autocast=None, progress_callback=ignore):
        checkpoint_metadata = self.checkpoint.metadata
        if autocast is None:
            autocast = checkpoint_metadata.precision

        if autocast is None:
            LOG.warning("No autocast given, using float16")
            autocast = "16"

        autocast = AUTOCAST[autocast]
        with Timer(f"Loading {self.checkpoint}"):
            try:
                model = torch.load(self.checkpoint.path, map_location=device, weights_only=False).to(device)
            except Exception:
                checkpoint_metadata.report_loading_error()
                raise

        model.eval()

        torch.set_grad_enabled(False)

        input_tensor_torch = torch.from_numpy(
            np.swapaxes(
                input_tensor_numpy,
                -2,
                -1,
            )[np.newaxis, ...]
        ).to(device)

        LOG.info("Using autocast %s", autocast)

        lead_time = frequency_to_timedelta(lead_time)
        steps = lead_time // self.frequency

        LOG.info("Lead time: %s, frequency: %s Forecasting %s steps", lead_time, self.frequency, steps)

        # output_tensor_index_to_variable = checkpoint_metadata.output_tensor_index_to_variable
        # prognostic_output_mask = checkpoint_metadata.prognostic_output_mask
        # prognostic_input_mask = checkpoint_metadata.prognostic_input_mask
        # computed_forcing_mask = checkpoint_metadata.computed_forcing_mask

        result = input_state.copy()
        result["fields"] = dict()
        print(result.keys())

        start_datetime = (
            max(input_state["dates"]) if isinstance(input_state["dates"], (list, tuple)) else input_state["dates"]
        )

        for i in range(steps):
            step = (i + 1) * self.frequency
            date = start_datetime + step
            LOG.info("Forecasting step %s (%s)", step, date)

            result["dates"] = [date]

            # Predict next state of atmosphere
            with torch.autocast(device_type=device, dtype=autocast):
                y_pred = model.predict_step(input_tensor_torch)

            # Detach tensor and squeeze (should we detach here?)
            output = np.squeeze(y_pred.cpu().numpy())  # shape: (values, variables)

            for i in range(output.shape[1]):
                result["fields"][checkpoint_metadata.output_tensor_index_to_variable[i]] = output[:, i]

            prognostic_fields = y_pred[..., checkpoint_metadata.prognostic_output_mask]

            # Compute new forcing

            # Update dynamic tensor for next iteration
            input_tensor_torch = input_tensor_torch.roll(-1, dims=1)
            input_tensor_torch[:, -1, :, checkpoint_metadata.prognostic_input_mask] = prognostic_fields

            # if checkpoint_metadata.computed_forcing_mask:

            #     forcing = forcing_and_constants(
            #         latitudes=input_state["latitudes"],
            #         longitudes=input_state["longitudes"],
            #         param=checkpoint_metadata.computed_time_dependent_forcings,
            #         date=date,
            #     )
            #     forcing = np.swapaxes(forcing[np.newaxis, np.newaxis, ...], -2, -1)
            #     forcing = torch.from_numpy(forcing).to(device)
            #     input_tensor_torch[:, -1, :, checkpoint_metadata.computed_forcing_mask] = forcing

            # progress_callback(i)

    @cached_property
    def frequency(self):
        return self.checkpoint.metadata.frequency

    @cached_property
    def lagged(self):
        result = list(range(0, self.checkpoint.metadata.multi_step_input))
        result = [-s * self.frequency for s in result]
        return sorted(result)

    def _report_mismatch(self, name, mask, input_fields, input_fields_numpy):
        LOG.error("Mismatch in %s and input_fields", name)
        LOG.error("%s: %s shape=(variables)", name, mask.shape)
        LOG.error("input_fields_numpy: %s shape=(dates, variables, grid)", input_fields_numpy.shape)
        LOG.error("MASK : %s", [self.checkpoint.metadata.variables[_] for _ in mask])

        # remapping = self.checkpoint.metadata.select["remapping"]
        # names = list(remapping.keys())

        # LOGGER.error(
        #     "INPUT: %s", [input_fields[i].metadata(*names, remapping=remapping) for i in range(len(input_fields) // 2)]
        # )
        raise ValueError(f"Mismatch in {name} and input_fields")

    def _report(self, name, mask):
        LOG.info("%s: %s", name, [self.checkpoint.metadata.variables[_] for _ in mask])
        LOG.info("%s: (%s items)", name, len(mask))

    def filter_and_sort(self, data, dates):
        typed_variables = self.checkpoint.typed_variables
        diagnostic_variables = self.checkpoint.diagnostic_variables

        def _name(field, key, original_metadata):
            warnings.warn("TEMPORARY CODE: Use the remapping in the metadata")
            param, levelist = original_metadata.get("param"), original_metadata.get("levelist")
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


class DefaultRunner(Runner):
    """_summary_

    Parameters
    ----------
    Runner : _type_
        _description_
    """

    pass
