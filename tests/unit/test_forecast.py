# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta

from anemoi.inference.runner import Runner
from anemoi.inference.tensors import TensorHandler

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


def basic_predict_step(model, input_tensors, **kwargs):
    input_tensor = input_tensors["data"]
    input_force = input_tensor[..., 0]  # (batch, multi_step_input, n_grid,)
    input_prog = input_tensor[..., 1]  # (batch, multi_step_input, n_grid,)
    multi_step_output = model.checkpoint.multi_step_output
    multi_step_input = model.checkpoint.multi_step_input
    assert multi_step_input == input_tensor.shape[1], "number of input steps mismatch"
    output_mask = model.checkpoint.output_mask

    for j in range(1, multi_step_input):
        assert input_prog[0, j] - input_prog[0, j - 1] == pytest.approx(torch.tensor([1.0, 1.0])), "prog step mismatch"
        assert input_force[0, j - 1] / input_force[0, j] == pytest.approx(
            torch.tensor([2.0, 2.0])
        ), "force step mismatch"
        assert input_force[0, j] == pytest.approx(0.5 ** (input_prog[0, j])), "force out of sync with prog "

    output_prog = torch.full((input_tensor.shape[0], multi_step_output, input_tensor.shape[2]), np.nan)

    for i in range(multi_step_output):
        output_prog[:, i, output_mask] = input_prog[:, -1, output_mask] + 1 + i

    diag = -output_prog  # (batch, multi_step_output, n_grid)

    output = torch.stack([output_prog, diag], dim=-1)  # (batch, multi_step_output, n_grid, n_vars_out)

    return dict(data=output.unsqueeze(2))


def _build_runner(metadata, dynamic_forcers, boundary_forcers):
    """Wire a bare Runner + TensorHandler around a mock metadata and its forcers."""
    runner = Runner.__new__(Runner)

    class TrivialModel(SimpleNamespace):
        def eval(self):
            pass

    runner._checkpoint = metadata  # single-dataset only, but sufficient for these tests
    runner.model = TrivialModel(checkpoint=metadata)
    runner.device = torch.device("cpu")
    runner.autocast = torch.bfloat16
    runner.verbosity = 0
    runner.use_profiler = False
    runner.hacks = None

    handler = TensorHandler.__new__(TensorHandler)
    handler.context = runner
    handler.metadata = metadata
    handler.trace = False
    handler._input_kinds = {}
    handler._input_tensor_by_name = ["force", "prog"]
    handler.dynamic_forcings_providers = dynamic_forcers
    handler.boundary_forcings_providers = boundary_forcers

    runner.tensor_handlers = dict(data=handler)
    return runner


@pytest.fixture
def forecast_runner_factory():
    def make_forecast_runner(multi_step_input=1, multi_step_output=1):
        timestep = timedelta(hours=1)
        metadata = SimpleNamespace(
            timestep=timestep,
            multi_step_input=multi_step_input,
            multi_step_output=multi_step_output,
            output_offsets=[i * timestep for i in range(1, multi_step_output + 1)],
            rollout_shift=multi_step_output * timestep,
            advance_map={
                "outin": [
                    (multi_step_output - i - 1, multi_step_input - i - 1)
                    for i in range(min(multi_step_output, multi_step_input))
                ],
                "inin": [(i, i - multi_step_output) for i in range(multi_step_output, multi_step_input)],
            },
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

        class GeometricDynamicForcer:
            mask = np.array([0])
            kinds = {}

            def load_forcings_array(self, dates, state):
                actual_step = round(state["step"] / metadata.timestep)
                n_dates = len(dates)
                values = np.array(
                    [np.float32(0.5 ** (actual_step - (n_dates - 1 - i))) for i in range(n_dates)],
                    dtype=np.float32,
                )
                return np.broadcast_to(values[np.newaxis, :, np.newaxis], (1, n_dates, 2)).copy()

        class SequentialBoundaryForcer:
            spatial_mask = ~metadata.output_mask
            variables_mask = metadata.prognostic_input_mask
            kinds = dict(retrieved=True)

            def load_forcings_array(self, dates, state):
                actual_step = round(state["step"] / metadata.timestep)
                n_dates = len(dates)
                values = np.array(
                    [np.float32((actual_step - (n_dates - 1 - i))) for i in range(n_dates)],
                    dtype=np.float32,
                )
                return np.broadcast_to(values[np.newaxis, :, np.newaxis], (1, n_dates, 1)).copy()

        return _build_runner(metadata, [GeometricDynamicForcer()], [SequentialBoundaryForcer()])

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
    monkeypatch.setattr(runner, "output_states_hook", lambda x: None)
    monkeypatch.setattr(runner, "mid_processors", defaultdict(list), raising=False)

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
        input_tensors_numpy=dict(data=test_input),
        input_states=dict(data={"date": datetime(2020, 1, 1)}),
    ):
        results_prog.append(new_state["data"]["fields"]["prog"][0].numpy())
        results_diag.append(new_state["data"]["fields"]["diag"][0].numpy())

    assert np.array(results_prog) == pytest.approx(expected_prog_output, abs=1e-4), "prog mismatch"
    assert np.array(results_diag) == pytest.approx(expected_diag_output, abs=1e-4), "diag mismatch"


# ── Offset-based rollout ───────────────────────────────────
# Same 2-grid force+prog world as above, but now with iregular input and output such as
# supported by the OffsetForecaster. Truth is time-based: prog(t) = t, force(t) = 0.5**t,
# with t in units of `_TIMESTEP` measured from the forecast start `_START`.

_START = datetime(2020, 1, 1)
_TIMESTEP = timedelta(hours=1)


def _t(date: datetime) -> float:
    return (date - _START) / _TIMESTEP


class _TimeDynamicForcer:  # force(t) = 0.5**t
    mask = np.array([0])
    kinds: dict = {}

    def load_forcings_array(self, dates, state):
        values = np.array([np.float32(0.5 ** _t(d)) for d in dates], dtype=np.float32)
        return np.broadcast_to(values[np.newaxis, :, np.newaxis], (1, len(dates), 2)).copy()


class _TimeBoundaryForcer:  # prog(t) = t on the boundary point
    spatial_mask = np.array([False, True])
    variables_mask = np.array([1])
    kinds = dict(retrieved=True)

    def load_forcings_array(self, dates, state):
        values = np.array([np.float32(_t(d)) for d in dates], dtype=np.float32)
        return np.broadcast_to(values[np.newaxis, :, np.newaxis], (1, len(dates), 1)).copy()


def offset_predict_step(model, input_tensors, **kwargs):
    input_tensor = input_tensors["data"]
    input_force = input_tensor[..., 0]
    input_prog = input_tensor[..., 1]
    assert torch.allclose(input_force, 0.5**input_prog, atol=1e-4), "force out of sync with prog"

    output_mask = model.checkpoint.output_mask
    steps = [offset / _TIMESTEP for offset in model.checkpoint.output_offsets]
    output_prog = torch.full((input_tensor.shape[0], len(steps), input_tensor.shape[2]), np.nan)
    for i, s in enumerate(steps):
        output_prog[:, i, output_mask] = input_prog[:, -1, output_mask] + s  # prog at (last input) + offset

    output = torch.stack([output_prog, -output_prog], dim=-1)  # (batch, mso, n_grid, n_vars_out)
    return dict(data=output.unsqueeze(2))


@pytest.fixture
def offset_forecast_runner_factory():
    def make(input_offsets, output_offsets, rollout_shift, advance_map):
        input_offsets = [to_timedelta(o) for o in input_offsets]
        metadata = SimpleNamespace(
            timestep=_TIMESTEP,
            multi_step_input=len(input_offsets),
            multi_step_output=len(output_offsets),
            output_offsets=[to_timedelta(o) for o in output_offsets],
            rollout_shift=to_timedelta(rollout_shift),
            advance_map=advance_map,
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
        runner = _build_runner(metadata, [_TimeDynamicForcer()], [_TimeBoundaryForcer()])
        return runner, input_offsets

    return make


@pytest.mark.parametrize(
    "input_offsets, output_offsets, rollout_shift, advance_map",
    [
        # Different input/output frequencies: 3h output at +3h is emitted but not fed back.
        pytest.param(["-6h", "0h"], ["3h", "6h"], "6h", {"inin": [(1, 0)], "outin": [(1, 1)]}, id="mixed-frequency"),
        # Multiple valid shifts, default (largest): feedback from +3h, +2h is emit-only, leaves gaps.
        pytest.param(["0h"], ["2h", "3h"], "3h", {"inin": [], "outin": [(1, 0)]}, id="multi-shift-default"),
        # Same offsets, explicit smaller shift: feedback from +2h instead.
        pytest.param(["0h"], ["2h", "3h"], "2h", {"inin": [], "outin": [(0, 0)]}, id="multi-shift-explicit"),
        # Irregular input offsets: the -1h slot is dropped on advance (needs the gather map).
        pytest.param(
            ["-4h", "-2h", "-1h", "0h"],
            ["1h", "2h"],
            "2h",
            {"inin": [(1, 0), (3, 1)], "outin": [(0, 2), (1, 3)]},
            id="irregular-offsets",
        ),
    ],
)
def test_forecast_offset(
    monkeypatch: pytest.MonkeyPatch,
    offset_forecast_runner_factory,
    input_offsets: list[str],
    output_offsets: list[str],
    rollout_shift: str,
    advance_map: dict,
):
    lead_time_hours = 12
    runner, input_offsets_td = offset_forecast_runner_factory(input_offsets, output_offsets, rollout_shift, advance_map)
    monkeypatch.setattr(runner, "predict_step", offset_predict_step)
    monkeypatch.setattr(runner, "output_states_hook", lambda x: None)
    monkeypatch.setattr(runner, "mid_processors", defaultdict(list), raising=False)

    # First input tensor: prog(t) = t, force(t) = 0.5**t at each input offset.
    input_steps = np.array([o / _TIMESTEP for o in input_offsets_td], dtype=np.float32)
    test_input_prog = np.broadcast_to(input_steps[:, np.newaxis], (len(input_steps), 2))
    test_input_force = np.broadcast_to((0.5**input_steps)[:, np.newaxis], (len(input_steps), 2))
    test_input = np.stack([test_input_force, test_input_prog], axis=1)  # (msi, n_vars=2, n_grid=2)

    # Expected forecasts: every valid time t = s*shift + offset (t <= lead), prog=t, diag=-t.
    shift_hours = to_timedelta(rollout_shift) / _TIMESTEP
    output_steps = [to_timedelta(o) / _TIMESTEP for o in output_offsets]
    expected = {}
    s = 0
    while s * shift_hours + min(output_steps) <= lead_time_hours:
        for o in output_steps:
            t = s * shift_hours + o
            if t <= lead_time_hours:
                expected[round(t)] = t
        s += 1

    results_prog = {}
    results_diag = {}
    for new_state in runner.forecast(
        lead_time=to_timedelta(f"{lead_time_hours}h"),
        input_tensors_numpy=dict(data=test_input),
        input_states=dict(data={"date": _START}),
    ):
        key = round(new_state["data"]["step"] / _TIMESTEP)
        results_prog[key] = new_state["data"]["fields"]["prog"][0].numpy()
        results_diag[key] = new_state["data"]["fields"]["diag"][0].numpy()

    assert sorted(results_prog) == sorted(expected), "forecast valid times mismatch"
    for key, t in expected.items():
        assert results_prog[key] == pytest.approx(np.float32(t), abs=1e-4), f"prog at t={t}"
        assert results_diag[key] == pytest.approx(np.float32(-t), abs=1e-4), f"diag at t={t}"
