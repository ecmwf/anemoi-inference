# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import datetime
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta

from anemoi.inference.runner import Runner

# This is a test to test stepping of the forecast method, in particular its robustness to changes in multi_step_ in- and output.
# The test checks if the forecast method of runner.py can correctly reproduce the "truth" described below.
# Say time runs in steps labelled by i and the origin i=0 corresponds to the last of the input steps
# The "true" fields then correspond to
# prog[i] = i, i.e.  -multi_step_input+1, ..., -1, 0, 1, 2, ..., lead_time_hours
# diag[i] = -i, i.e. multi_step_input-1, ..., 1, 0, -1, -2, ..., -lead_time_hours
# force[i] = 0.5**i, i.e. (1/2)** (-multi_step_input+1), ..., 2, 1, 1/2, 1/4, ..., (1/2) ** lead_time_hours
#
# Predict step does the following:
#  1. checks prog input and force input,
#  2. moves prog forward,
#  3. creates diag
#  4. produces nan on the boundary
#
# The setup, see forecast_runner_factory, is:
#  timestep of 1h, multi_step_input and multi_step_output are parameters
#  2 grid points: index 0 interior, index 1 boundary.
#  3 variables: force (index 0 in input), prog ( index 1 in input, index 0 in output), diag (index 1 in output).
#  Dynamic forcings:   forcing field with value 0.5**i
#  Boundary forcings:  overwrite values at boundary points with prog


def basic_predict_step(model, input_tensor, **kwargs):
    input_force = input_tensor[..., 0]  # (batch, multi_step_input, n_grid,)
    input_prog = input_tensor[..., 1]  # (batch, multi_step_input, n_grid,)
    multi_step_output = model.checkpoint.multi_step_output
    multi_step_input = model.checkpoint.multi_step_input
    assert multi_step_input == input_tensor.shape[1], "number of input steps mismatch"
    output_mask = model.checkpoint.output_mask

    for j in range(1, multi_step_input):
        assert input_prog[0, j] - input_prog[0, j - 1] == pytest.approx(torch.tensor([1.0, 1.0])), "prog step mismatch"
        assert input_force[0, j - 1, 0] / input_force[0, j, 0] == pytest.approx(2.0), "force step mismatch"

    output_prog = torch.full((input_tensor.shape[0], multi_step_output, input_tensor.shape[2]), np.nan)

    for i in range(multi_step_output):
        output_prog[:, i, output_mask] = input_prog[:, -1, output_mask] + 1 + i

    diag = -output_prog  # (batch, multi_step_output, n_grid)

    output = torch.stack([output_prog, diag], dim=-1)  # (batch, multi_step_output, n_grid, n_vars_out)

    return output.unsqueeze(2)


@pytest.fixture
def forecast_runner_factory():
    def make_forecast_runner(multi_step_input=1, multi_step_output=1):

        runner = Runner.__new__(Runner)

        runner._checkpoint = SimpleNamespace(
            timestep=timedelta(hours=1),
            multi_step_input=multi_step_input,
            multi_step_output=multi_step_output,
            variable_to_input_tensor_index={"force": 0, "prog": 1},
            output_tensor_index_to_variable=["prog", "diag"],
            typed_variables={
                "force": SimpleNamespace(is_constant_in_time=False),
                "prog": SimpleNamespace(is_constant_in_time=False),
            },
            prognostic_input_mask=np.array([1]),
            prognostic_output_mask=np.array([0]),
            output_mask=np.array([True, False]),
        )

        class TrivialModel(SimpleNamespace):
            def eval(self):
                pass

        runner.model = TrivialModel(checkpoint=runner._checkpoint)
        runner.device = torch.device("cpu")
        runner.verbosity = 0
        runner.trace = False
        runner.use_profiler = False
        runner.autocast = torch.float32
        runner._input_tensor_by_name = ["force", "prog"]
        runner._input_kinds = {}
        runner.hacks = None

        checkpoint = runner._checkpoint

        class GeometricDynamicForcer:
            mask = np.array([0])
            kinds = {}

            def load_forcings_array(self, dates, state):
                actual_step = round(state["step"] / checkpoint.timestep)
                n_dates = len(dates)
                values = np.array(
                    [np.float32(0.5 ** (actual_step - (n_dates - 1 - i))) for i in range(n_dates)],
                    dtype=np.float32,
                )
                return np.broadcast_to(values[np.newaxis, :, np.newaxis], (1, n_dates, 2)).copy()

        runner.dynamic_forcings_inputs = [GeometricDynamicForcer()]

        class SequentialBoundaryForcer:
            spatial_mask = ~checkpoint.output_mask
            variables_mask = checkpoint.prognostic_input_mask
            kinds = dict(retrieved=True)

            def load_forcings_array(self, dates, state):
                actual_step = round(state["step"] / checkpoint.timestep)
                n_dates = len(dates)
                values = np.array(
                    [np.float32((actual_step - (n_dates - 1 - i))) for i in range(n_dates)],
                    dtype=np.float32,
                )
                return np.broadcast_to(values[np.newaxis, :, np.newaxis], (1, n_dates, 1)).copy()

        runner.boundary_forcings_inputs = [SequentialBoundaryForcer()]

        return runner

    return make_forecast_runner


@pytest.mark.parametrize(
    "multi_step_input, multi_step_output, lead_time_hours",
    [
        pytest.param(1, 1, 5, id="in1-out1"),
        pytest.param(1, 2, 5, id="in1-out2"),
        pytest.param(1, 3, 5, id="in1-out3"),
        pytest.param(2, 1, 5, id="in2-out1"),
        pytest.param(2, 2, 5, id="in2-out2"),
        pytest.param(2, 3, 5, id="in2-out3"),
        pytest.param(3, 1, 5, id="in3-out1"),
        pytest.param(3, 2, 5, id="in3-out2"),
        pytest.param(3, 3, 5, id="in3-out3"),
    ],
)
def test_forecast(
    monkeypatch: pytest.MonkeyPatch,
    forecast_runner_factory,
    multi_step_input: int,
    multi_step_output: int,
    lead_time_hours: int,
):
    runner = forecast_runner_factory(multi_step_input, multi_step_output)
    monkeypatch.setattr(runner, "predict_step", basic_predict_step)
    monkeypatch.setattr(runner, "output_state_hook", lambda x: None)

    lead_time = to_timedelta(f"{lead_time_hours}h")

    input_steps = np.arange(1 - multi_step_input, 1, dtype=np.float32)  # [-msi+1, ..., 0]
    test_input_prog = np.broadcast_to(input_steps[:, np.newaxis], (multi_step_input, 2))
    test_input_force = np.broadcast_to((0.5**input_steps)[:, np.newaxis], (multi_step_input, 2))
    # stack in variable dim (force=0, prog=1) → (msi, n_vars=2, n_grid=2)
    test_input = np.stack([test_input_force, test_input_prog], axis=1)

    expected_prog_output = np.arange(1, lead_time_hours + 1, dtype=np.float32)
    expected_diag_output = -expected_prog_output

    results_prog = []
    results_diag = []
    for new_state in runner.forecast(
        lead_time=lead_time,
        input_tensor_numpy=test_input,
        input_state={"date": datetime(2020, 1, 1)},
    ):
        results_prog.append(new_state["fields"]["prog"][0].numpy())
        results_diag.append(new_state["fields"]["diag"][0].numpy())

    assert np.array(results_prog) == pytest.approx(expected_prog_output, abs=1e-4), "prog mismatch"
    assert np.array(results_diag) == pytest.approx(expected_diag_output, abs=1e-4), "diag mismatch"
