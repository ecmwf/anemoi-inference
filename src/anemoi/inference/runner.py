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
from anemoi.utils.timer import Timer

from .checkpoint import Checkpoint

LOGGER = logging.getLogger(__name__)


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


class Runner:
    """_summary_"""

    def __init__(self, checkpoint):
        self.checkpoint = Checkpoint(checkpoint)

    def run(
        self,
        *,
        input_fields,
        lead_time,
        device,
        start_datetime=None,
        output_callback=ignore,
        autocast=None,
        progress_callback=ignore,
    ) -> None:
        """_summary_

        Parameters
        ----------
        input_fields : _type_
            _description_
        lead_time : _type_
            _description_
        device : _type_
            _description_
        start_datetime : _type_, optional
            _description_, by default None
        output_callback : _type_, optional
            _description_, by default ignore
        autocast : _type_, optional
            _description_, by default None
        progress_callback : _type_, optional
            _description_, by default ignore

        Raises
        ------
        RuntimeError
            _description_
        ValueError
            _description_
        """

        self.checkpoint.summary()

        if autocast is None:
            autocast = self.checkpoint.precision

        if autocast is None:
            LOGGER.warning("No autocast given, using float16")
            autocast = "16"

        autocast = AUTOCAST[autocast]

        input_fields = input_fields.sel(**self.checkpoint.select)
        input_fields = input_fields.order_by(**self.checkpoint.order_by)

        number_of_grid_points = len(input_fields[0].grid_points()[0])

        LOGGER.info("Loading input: %d fields (lagged=%d)", len(input_fields), len(self.lagged))

        if start_datetime is None:
            start_datetime = input_fields.order_by(valid_datetime="ascending")[-1].datetime()["valid_time"]

        num_fields_per_date = len(input_fields) // len(self.lagged)  # assumed

        # Check valid_datetime of input data
        # The subsequent reshape operation assumes that input_fields are chunkable by datetime
        for i, lag in enumerate(self.lagged):
            date = start_datetime + datetime.timedelta(hours=lag)
            dates_found = set(
                field.datetime()["valid_time"]
                for field in input_fields[i * num_fields_per_date : (i + 1) * num_fields_per_date]
            )
            # All chunks must have the same datetime that must agree with the lag
            if dates_found != {date}:
                raise RuntimeError(
                    "Inconsistent datetimes detected.\n"
                    f"Datetimes in data: {', '.join(d.isoformat() for d in dates_found)}.\n"
                    f"Expected datetime: {date.isoformat()} (for lag {lag})"
                )

        input_fields_numpy = input_fields.to_numpy(dtype=np.float32, flatten=True)

        input_fields_numpy = input_fields_numpy.reshape(
            len(self.lagged),
            num_fields_per_date,
            number_of_grid_points,
        )  # nlags, nparams, ngrid

        # Used to check if we cover the whole input, with no overlaps
        check = np.full(self.checkpoint.num_input_features, fill_value=False, dtype=np.bool_)

        kinds = np.full(self.checkpoint.num_input_features, fill_value="?", dtype=np.character)

        inputs = np.full(self.checkpoint.num_input_features, fill_value=False, dtype=np.bool_)

        # E.g cos_latitude
        computed_constant_mask = self.checkpoint.computed_constants_mask
        assert not np.any(check[computed_constant_mask]), check
        check[computed_constant_mask] = True
        kinds[computed_constant_mask] = "C"

        # E.g. lsm, orography
        constant_from_input_mask = self.checkpoint.constants_from_input_mask
        assert not np.any(check[constant_from_input_mask]), check
        check[constant_from_input_mask] = True
        kinds[constant_from_input_mask] = "K"
        inputs[constant_from_input_mask] = True

        # e.g. isolation
        computed_forcing_mask = self.checkpoint.computed_forcings_mask
        assert not np.any(check[computed_forcing_mask]), check
        check[computed_forcing_mask] = True
        kinds[computed_forcing_mask] = "F"

        # e.g 2t, 10u, 10v
        prognostic_input_mask = self.checkpoint.prognostic_input_mask
        assert not np.any(check[prognostic_input_mask]), check
        check[prognostic_input_mask] = True
        kinds[prognostic_input_mask] = "P"
        inputs[prognostic_input_mask] = True

        #
        if not np.all(check):
            for i, c in enumerate(check):
                if not c:
                    LOGGER.error(
                        "Missing %s %s %s",
                        i,
                        self.checkpoint.model_to_data[i],
                        self.checkpoint.index_to_variable[self.checkpoint.model_to_data[i]],
                    )
            raise RuntimeError("Missing fields")

        prognostic_data_from_retrieved_fields_mask = []
        constant_data_from_retrieved_fields_mask = []

        MARS = {False: " ", True: "X"}

        retrieved_fields_index = 0
        for i, c in enumerate(check):
            if inputs[i]:
                assert kinds[i].decode() in ("P", "K")
                if kinds[i].decode() == "P":
                    prognostic_data_from_retrieved_fields_mask.append(retrieved_fields_index)
                else:
                    constant_data_from_retrieved_fields_mask.append(retrieved_fields_index)
                retrieved_fields_index += 1

            if hasattr(self, "verbose") and self.verbose:
                print(
                    "{:4d} {:1s} {} {:4d} {:10s}".format(
                        i,
                        kinds[i].decode(),
                        MARS[inputs[i]],
                        self.checkpoint.model_to_data[i],
                        self.checkpoint.index_to_variable[self.checkpoint.model_to_data[i]],
                    )
                )

        prognostic_data_from_retrieved_fields_mask = np.array(prognostic_data_from_retrieved_fields_mask)
        constant_data_from_retrieved_fields_mask = np.array(constant_data_from_retrieved_fields_mask)

        # Build the input tensor

        input_tensor_numpy = np.full(
            shape=(
                len(self.lagged),
                self.checkpoint.num_input_features,
                number_of_grid_points,
            ),
            fill_value=np.nan,
            dtype=np.float32,
        )  # nlags, nparams, ngrid

        # Check that the computed constant mask and the constant from input mask are disjoint
        # assert np.amax(prognostic_input_mask) < np.amin(constant_from_input_mask)

        input_tensor_numpy[:, prognostic_input_mask] = input_fields_numpy[:, prognostic_data_from_retrieved_fields_mask]

        input_tensor_numpy[:, constant_from_input_mask] = input_fields_numpy[
            :, constant_data_from_retrieved_fields_mask
        ]

        constants = forcing_and_constants(
            source=input_fields[:1],
            param=self.checkpoint.computed_constants,
            date=start_datetime,
        )

        for i in range(len(self.lagged)):
            input_tensor_numpy[i, computed_constant_mask] = constants

        for i in range(len(self.lagged)):
            forcings = forcing_and_constants(
                source=input_fields[:1],
                param=self.checkpoint.computed_forcings,
                date=start_datetime + datetime.timedelta(hours=self.lagged[i]),
            )
            input_tensor_numpy[i, computed_forcing_mask] = forcings

        LOGGER.info("Input tensor shape: %s", input_tensor_numpy.shape)

        imputable_variables = self.checkpoint.imputable_variables
        can_be_missing = set()
        # check for NaNs
        for i in range(input_tensor_numpy.shape[1]):
            name = self.checkpoint.index_to_variable[self.checkpoint.model_to_data[i]]
            has_missing = np.isnan(input_tensor_numpy[:, i, :]).any()
            is_imputable = name in imputable_variables
            if has_missing:
                can_be_missing.add(name)
                if not is_imputable:
                    model_index = self.checkpoint.model_to_data[i]
                    LOGGER.error(
                        "No imputation specified for NaNs in %s (%s %s)",
                        name,
                        i,
                        model_index,
                    )
                    raise ValueError(f"Field '{name}' has NaNs and is not marked as imputable")

        with Timer(f"Loading {self.checkpoint}"):
            try:
                model = torch.load(self.checkpoint.path, map_location=device, weights_only=False).to(device)
            except Exception:
                self.checkpoint.report_loading_error()
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

        prognostic_output_mask = self.checkpoint.prognostic_output_mask
        diagnostic_output_mask = self.checkpoint.diagnostic_output_mask

        LOGGER.info("Using autocast %s", autocast)

        # Write dynamic fields
        def get_most_recent_datetime(input_fields):
            datetimes = [f.datetime()["valid_time"] for f in input_fields]
            latest = datetimes[-1]
            for d in datetimes:
                assert d <= latest, (datetimes, d, latest)
            return latest

        most_recent_datetime = get_most_recent_datetime(input_fields)
        reference_fields = [f for f in input_fields if f.datetime()["valid_time"] == most_recent_datetime]
        prognostic_template = reference_fields[self.checkpoint.variable_to_index["lsm"]]

        accumulated_output = np.zeros(
            shape=(len(diagnostic_output_mask), number_of_grid_points),
            dtype=np.float32,
        )

        if self.checkpoint.diagnostic_params:
            output_callback(
                input_fields,
                self.checkpoint.diagnostic_params,
                prognostic_template,
                accumulated_output[0].shape,
            )
        else:
            output_callback(input_fields)

        prognostic_params = self.checkpoint.prognostic_params
        accumulations_params = self.checkpoint.accumulations_params

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
                for n, param in enumerate(self.checkpoint.diagnostic_params):
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
                param=self.checkpoint.computed_forcings,
                date=start_datetime + datetime.timedelta(hours=step),
            )
            forcing = np.swapaxes(forcing[np.newaxis, np.newaxis, ...], -2, -1)
            forcing = torch.from_numpy(forcing).to(device)

            # Update dynamic tensor for next iteration
            input_tensor_torch = input_tensor_torch.roll(-1, dims=1)
            input_tensor_torch[:, -1, :, prognostic_input_mask] = prognostic_fields
            input_tensor_torch[:, -1, :, computed_forcing_mask] = forcing

            # progress_callback(i)

    @cached_property
    def hour_steps(self):
        return self.checkpoint.hour_steps

    @cached_property
    def lagged(self):
        result = list(range(0, self.checkpoint.multi_step))
        result = [-s * self.hour_steps for s in result]
        return sorted(result)

    @property
    def param_sfc(self):
        param_sfc = self.checkpoint.param_sfc

        # Remove diagnostic params

        param_sfc = [p for p in param_sfc if p not in self.checkpoint.diagnostic_params]

        return param_sfc

    @property
    def param_level_pl(self):

        # To do remove diagnostic params

        return self.checkpoint.param_level_pl

    @property
    def param_level_ml(self):

        # To do remove diagnostic params

        return self.checkpoint.param_level_ml


class DefaultRunner(Runner):
    """_summary_

    Parameters
    ----------
    Runner : _type_
        _description_
    """

    pass
