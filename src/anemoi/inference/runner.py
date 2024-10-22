# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections import defaultdict
from functools import cached_property

import numpy as np
import torch

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


def forcing_and_constants(source, date, param):
    import earthkit.data as ekd

    ds = ekd.from_source(
        "forcings",
        source,
        date=date,
        param=param,
    )

    assert len(ds) == len(param), (len(ds), len(param), date)

    return ds.to_numpy(dtype=np.float32)


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
        input_fields = checkpoint_metadata.filter_and_sort(input_fields, dates)

        check = defaultdict(set)

        for field in input_fields:
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

        input_tensor_numpy = np.full(
            shape=(len(dates), checkpoint_metadata.num_input_features, checkpoint_metadata.number_of_grid_points),
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

        missing = set(range(checkpoint_metadata.num_input_features)) - check
        if missing:
            LOG.error("Missing variables %s", [checkpoint_metadata.model_input_variables[i] for i in missing])

            assert not missing, missing

        return input_tensor_numpy

    """
    def forecast(self):
        checkpoint_metadata = self.checkpoint.metadata
        if autocast is None:
            autocast = checkpoint_metadata.precision

        if autocast is None:
            LOG.warning("No autocast given, using float16")
            autocast = "16"

        autocast = AUTOCAST[autocast]
        with Timer(f"Loading {self.checkpoint}"):
            try:
                model = torch.load(checkpoint_metadata.path, map_location=device, weights_only=False).to(device)
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

        prognostic_output_mask = checkpoint_metadata.prognostic_output_mask
        diagnostic_output_mask = checkpoint_metadata.diagnostic_output_mask

        LOG.info("Using autocast %s", autocast)

        # Write dynamic fields
        def get_most_recent_datetime(input_fields):
            datetimes = [f.datetime()["valid_time"] for f in input_fields]
            latest = datetimes[-1]
            for d in datetimes:
                assert d <= latest, (datetimes, d, latest)
            return latest

        most_recent_datetime = get_most_recent_datetime(input_fields)
        reference_fields = [f for f in input_fields if f.datetime()["valid_time"] == most_recent_datetime]

        if "lsm" in checkpoint_metadata.variable_to_index:
            prognostic_template = reference_fields[checkpoint_metadata.variable_to_index["lsm"]]
        else:
            first = list(checkpoint_metadata.variable_to_index.keys())
            LOG.warning("No LSM found to use as a GRIB template, using %s", first[0])
            prognostic_template = reference_fields[0]

        accumulated_output = np.zeros(
            shape=(len(diagnostic_output_mask), number_of_grid_points),
            dtype=np.float32,
        )

        if checkpoint_metadata.diagnostic_params:
            output_callback(
                input_fields,
                checkpoint_metadata.diagnostic_params,
                prognostic_template,
                accumulated_output[0].shape,
            )
        else:
            output_callback(input_fields)

        prognostic_params = checkpoint_metadata.prognostic_params
        accumulations_params = checkpoint_metadata.accumulations_params

        # with self.stepper(self.hour_steps) as stepper:

        for i in progress_callback(range(lead_time // self.hour_steps)):
            step = (i + 1) * self.hour_steps

            # Predict next state of atmosphere
            with torch.autocast(device_type=device, dtype=autocast):
                y_pred = model.predict_step(input_tensor_torch)

            # Detach tensor and squeeze
            output = np.squeeze(y_pred.cpu().numpy())

            prognostic_fields_numpy = output[:, prognostic_output_mask]
            if len(diagnostic_output_mask):
                diagnostic_fields_numpy = output[:, diagnostic_output_mask]

            for n, (m, param) in enumerate(zip(prognostic_data_from_retrieved_fields_mask, prognostic_params)):
                template = reference_fields[m]
                assert template.datetime()["valid_time"] == most_recent_datetime, (
                    template.datetime()["valid_time"],
                    most_recent_datetime,
                )
                output_callback(
                    prognostic_fields_numpy[:, n],
                    template=template,
                    step=step,
                    check_nans=True,  # param in can_be_missing,
                )

            # Write diagnostic fields
            if len(diagnostic_output_mask):
                for n, param in enumerate(checkpoint_metadata.diagnostic_params):
                    accumulated_output[n] += np.maximum(0, diagnostic_fields_numpy[:, n])
                    assert prognostic_template.datetime()["valid_time"] == most_recent_datetime, (
                        prognostic_template.datetime()["valid_time"],
                        most_recent_datetime,
                    )

                    if param in accumulations_params:
                        output_callback(
                            accumulated_output[n],
                            stepType="accum",
                            template=prognostic_template,
                            startStep=0,
                            endStep=step,
                            param=param,
                            check_nans=True,  # param in can_be_missing,
                        )
                    else:
                        output_callback(
                            diagnostic_fields_numpy[:, n],
                            template=prognostic_template,
                            step=step,
                            check_nans=True,  # param in can_be_missing,
                        )

            # Next step

            prognostic_fields = y_pred[..., prognostic_output_mask]

            # Compute new forcing

            forcing = forcing_and_constants(
                source=input_fields[:1],
                param=checkpoint_metadata.computed_forcings,
                date=start_datetime + datetime.timedelta(hours=step),
            )
            forcing = np.swapaxes(forcing[np.newaxis, np.newaxis, ...], -2, -1)
            forcing = torch.from_numpy(forcing).to(device)

            # Update dynamic tensor for next iteration
            input_tensor_torch = input_tensor_torch.roll(-1, dims=1)
            input_tensor_torch[:, -1, :, prognostic_input_mask] = prognostic_fields
            if computed_forcing_mask:
                input_tensor_torch[:, -1, :, computed_forcing_mask] = forcing

            # progress_callback(i)

    """

    @cached_property
    def hour_steps(self):
        return self.checkpoint.metadata.hour_steps

    @cached_property
    def lagged(self):
        result = list(range(0, self.checkpoint.metadata.multi_step))
        result = [-s * self.hour_steps for s in result]
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


class DefaultRunner(Runner):
    """_summary_

    Parameters
    ----------
    Runner : _type_
        _description_
    """

    pass
