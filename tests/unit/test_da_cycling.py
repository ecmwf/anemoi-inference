# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The flow-dependent residual base supplied by the DA cycling runner.

Training feeds ``skip_input`` on exactly the first ``da_cycles + 1`` model calls
-- every DA cycle plus the first forecast step, whose input is the last analysis
-- and nothing after that. These pin the inference side to that same window and
to the same tensor contents.

Building a real runner needs a checkpoint, so these drive the runner methods
against stubs carrying only the attributes they read, following the
no-checkpoint pattern in ``test_forecast.py``.
"""

from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anemoi.inference.runners.da_cycling import DACyclingRunner

_GRID, _N_VARS = 4, 3
_PROG_IN = np.array([1])  # input columns: [force, prog, corr]
_PROG_OUT = np.array([0])
_CORR_IN = np.array([2], dtype=np.int64)


def _metadata(multi_step_input: int = 2, multi_step_output: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        timestep=timedelta(hours=6),
        multi_step_input=multi_step_input,
        multi_step_output=multi_step_output,
        prognostic_input_mask=_PROG_IN,
        prognostic_output_mask=_PROG_OUT,
        corrector_input_mask=_CORR_IN,
        variable_to_input_tensor_index={"force": 0, "prog": 1, "corr": 2},
    )


def _runner(da_cycles: int = 2, flow_dependent: bool = True, datasets=("data",)) -> DACyclingRunner:
    """A runner stub exposing only what the skip-base machinery reads."""
    runner = DACyclingRunner.__new__(DACyclingRunner)
    runner.da_cycles = da_cycles
    runner.da_flow_dependent_skip = flow_dependent
    runner._pending_skip_input = None
    runner.tensor_handlers = {ds: SimpleNamespace(metadata=_metadata()) for ds in datasets}
    return runner


# ── the residual base itself ──────────────────────────────────────────────


def test_build_skip_input_undoes_the_observation_copy() -> None:
    runner = _runner()

    # A blended state: prognostic column holds the observation at grid 0 and the
    # model background elsewhere, exactly as _da_blend leaves it.
    blended = torch.zeros(1, 2, _GRID, _N_VARS)
    blended[0, -1, :, 1] = torch.tensor([5.0, 99.0, 99.0, 99.0])
    blended[0, -1, :, 0] = 7.0  # forcing
    blended[0, -1, :, 2] = 3.0  # corrector
    y_pred = torch.full((1, 1, 1, _GRID, 1), 99.0)  # the raw background

    base = runner._build_skip_input({"data": blended}, {"data": y_pred})["data"]

    # The residual base is the pure background everywhere: the observation at grid 0
    # is gone, so it can no longer reach the output through the additive path.
    assert torch.allclose(base[0, -1, :, 1], torch.full((_GRID,), 99.0))
    # ... while the state the encoder sees is untouched.
    assert torch.allclose(blended[0, -1, :, 1], torch.tensor([5.0, 99.0, 99.0, 99.0]))
    # Non-prognostic columns are carried through: only prognostics are consumed
    # downstream, and rewriting the rest would diverge from training.
    assert torch.allclose(base[0, -1, :, 0], torch.full((_GRID,), 7.0))
    assert torch.allclose(base[0, -1, :, 2], torch.full((_GRID,), 3.0))


def test_build_skip_input_leaves_history_slots_alone() -> None:
    # Only the freshly written slots carry a new background; earlier slots are
    # history that _da_blend already rolled into place.
    runner = _runner()
    blended = torch.rand(1, 2, _GRID, _N_VARS)
    y_pred = torch.full((1, 1, 1, _GRID, 1), 99.0)

    base = runner._build_skip_input({"data": blended}, {"data": y_pred})["data"]

    assert torch.equal(base[0, 0], blended[0, 0])


def test_build_skip_input_covers_every_dataset() -> None:
    # forward indexes skip_input[dataset_name] strictly, so a dataset missing from
    # the base -- e.g. one with no observation source -- would raise a KeyError.
    runner = _runner(datasets=("data", "no_obs"))
    tensors = {ds: torch.zeros(1, 2, _GRID, _N_VARS) for ds in ("data", "no_obs")}
    y_preds = {ds: torch.zeros(1, 1, 1, _GRID, 1) for ds in ("data", "no_obs")}

    assert set(runner._build_skip_input(tensors, y_preds)) == {"data", "no_obs"}


# ── the one-shot injection ────────────────────────────────────────────────


def _record_predict_step(runner, monkeypatch) -> list:
    """Capture the kwargs of every super().predict_step call."""
    seen = []

    def fake(self, model, input_tensors_torch, **kwargs):  # noqa: ARG001
        seen.append(kwargs)
        return {"data": torch.zeros(1, 1, 1, _GRID, 1)}

    monkeypatch.setattr("anemoi.inference.runners.default.DefaultRunner.predict_step", fake)
    return seen


def test_predict_step_injects_pending_base_once(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    seen = _record_predict_step(runner, monkeypatch)
    tensors = {"data": torch.zeros(1, 2, _GRID, _N_VARS)}

    runner._pending_skip_input = {"data": None}
    runner.predict_step(None, tensors)
    runner.predict_step(None, tensors)

    # One-shot: the base reaches the call it was staged for and no later one, so a
    # forecast step never silently inherits a DA cycle's residual base.
    assert seen[0]["skip_input"] == {"data": None}
    assert "skip_input" not in seen[1]
    assert runner._pending_skip_input is None


def test_predict_step_is_transparent_when_nothing_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner(flow_dependent=False)
    seen = _record_predict_step(runner, monkeypatch)

    runner.predict_step(None, {"data": torch.zeros(1, 2, _GRID, _N_VARS)}, fcstep=0)

    assert seen == [{"fcstep": 0}]


# ── the call window ───────────────────────────────────────────────────────


def _drive_cycles(runner, n_calls: int) -> list:
    """Replay the cycle loop's staging, returning what each call would receive.

    Reproduces the ordering in ``forecast``: stage the None base before the loop,
    then re-stage after each cycle's blend, and keep calling past the handoff.
    """
    seen = []
    tensors = {"data": torch.zeros(1, 2, _GRID, _N_VARS)}
    y_preds = {"data": torch.zeros(1, 1, 1, _GRID, 1)}

    if runner.da_flow_dependent_skip:
        runner._pending_skip_input = {ds: None for ds in runner.tensor_handlers}

    for call in range(n_calls):
        seen.append(runner._pending_skip_input)
        runner._pending_skip_input = None  # consumed by predict_step
        # The runner re-stages only while cycling; the base left by the final cycle
        # is what the first forecast step consumes.
        if call < runner.da_cycles and runner.da_flow_dependent_skip:
            runner._pending_skip_input = runner._build_skip_input(tensors, y_preds)

    return seen


@pytest.mark.parametrize("da_cycles", [1, 2, 4])
def test_skip_input_window_is_da_cycles_plus_one(da_cycles: int) -> None:
    """Training feeds skip_input on the first da_cycles + 1 calls, then stops."""
    runner = _runner(da_cycles=da_cycles)

    staged = [s is not None for s in _drive_cycles(runner, da_cycles + 3)]

    assert staged == [True] * (da_cycles + 1) + [False] * 2


def test_first_call_has_no_background_and_later_calls_do() -> None:
    runner = _runner(da_cycles=2)

    seen = _drive_cycles(runner, 3)

    # Cycle 0: no background exists yet, so None -> zeros after normalization.
    assert seen[0] == {"data": None}
    # Cycles 1+ and the handoff: a real background tensor.
    for staged in seen[1:]:
        assert isinstance(staged["data"], torch.Tensor)


def test_flag_off_never_stages_a_base() -> None:
    runner = _runner(da_cycles=2, flow_dependent=False)

    staged = _drive_cycles(runner, 4)

    assert staged == [None] * 4
