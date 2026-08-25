# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anemoi.inference.tensors import TensorHandler

# Tensor layout used throughout: (batch, multi_step_input, values, variables).
# Input variables:  index 0 = "force" (non-prognostic), index 1 = "prog" (prognostic).
# Output variables: index 0 = "prog" (prognostic),       index 1 = "diag" (diagnostic).
_N_GRID = 2


def _regular_advance_map(multi_step_input: int, multi_step_output: int) -> dict:
    """Reproduce ``Metadata.advance_map`` for a regular grid (time-shift by ``multi_step_output``)."""
    n, m = multi_step_input, multi_step_output
    return {
        "outin": [(m - i - 1, n - i - 1) for i in range(min(n, m))],
        "inin": [(i, i - m) for i in range(m, n)],
    }


def _make_handler(
    multi_step_input: int,
    multi_step_output: int,
    advance_map: dict,
    *,
    prognostic_input_mask: tuple[int, ...] = (1,),
    prognostic_output_mask: tuple[int, ...] = (0,),
    input_names: tuple[str, ...] = ("force", "prog"),
) -> TensorHandler:
    metadata = SimpleNamespace(
        dataset_name="data",
        multi_step_input=multi_step_input,
        multi_step_output=multi_step_output,
        prognostic_input_mask=np.array(prognostic_input_mask),
        prognostic_output_mask=np.array(prognostic_output_mask),
        advance_map=advance_map,
    )
    handler = TensorHandler.__new__(TensorHandler)
    handler.metadata = metadata
    handler.trace = False
    handler._input_kinds = {}
    handler._input_tensor_by_name = list(input_names)
    return handler


def _build_tensors(multi_step_input: int, multi_step_output: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode each input step ``s`` as value ``s + 1`` and each prediction ``o`` as ``n + o + 1``.

    With this encoding the concatenated prognostic sequence is ``[1, ..., n, n+1, ..., n+m]``,
    so advancing the window by ``m`` steps yields the prognostic column ``[m+1, ..., m+n]``.
    """
    n, m = multi_step_input, multi_step_output
    input_tensor = torch.zeros((1, n, _N_GRID, 2))
    for step in range(n):
        input_tensor[:, step, :, 0] = float(step + 1)  # force (stale placeholder)
        input_tensor[:, step, :, 1] = float(step + 1)  # prognostic
    y_pred = torch.zeros((1, m, _N_GRID, 2))
    for out in range(m):
        y_pred[:, out, :, 0] = float(n + out + 1)  # prognostic output
        y_pred[:, out, :, 1] = -float(n + out + 1)  # diagnostic output
    return input_tensor, y_pred


@pytest.mark.parametrize(
    ("multi_step_input", "multi_step_output", "expected_prog"),
    [
        pytest.param(1, 1, [2.0], id="in1-out1"),
        pytest.param(2, 1, [2.0, 3.0], id="in2-out1"),
        pytest.param(2, 2, [3.0, 4.0], id="in2-out2"),
        pytest.param(2, 3, [4.0, 5.0], id="in2-out3"),
        pytest.param(3, 1, [2.0, 3.0, 4.0], id="in3-out1"),
        pytest.param(3, 2, [3.0, 4.0, 5.0], id="in3-out2"),
        pytest.param(3, 3, [4.0, 5.0, 6.0], id="in3-out3"),
    ],
)
def test_copy_prognostic_fields_advances_window(
    multi_step_input: int,
    multi_step_output: int,
    expected_prog: list[float],
) -> None:
    """The prognostic window slides forward, filling the tail from model predictions."""
    handler = _make_handler(
        multi_step_input,
        multi_step_output,
        _regular_advance_map(multi_step_input, multi_step_output),
    )
    input_tensor, y_pred = _build_tensors(multi_step_input, multi_step_output)
    check = np.zeros(2, dtype=bool)

    result = handler.copy_prognostic_fields_to_input_tensor(input_tensor, y_pred, check)

    for step, value in enumerate(expected_prog):
        assert torch.all(result[0, step, :, 1] == value), (
            f"prognostic step {step} for in{multi_step_input}-out{multi_step_output} "
            f"should be {value}, got {result[0, step, :, 1].tolist()}"
        )
    # Only the prognostic input slot is flagged as written.
    assert check.tolist() == [False, True]
    assert handler._input_kinds["prog"].attributes == {"prognostic": True}


@pytest.mark.parametrize(
    ("multi_step_input", "multi_step_output", "advance_map", "expected_prog"),
    [
        # Irregular grids that a regular time-shift cannot represent: the tail input slot
        # is filled from a non-adjacent prediction (skipping earlier output steps).
        pytest.param(2, 2, {"inin": [(1, 0)], "outin": [(1, 1)]}, [2.0, 4.0], id="skip-out0"),
        pytest.param(2, 3, {"inin": [(1, 0)], "outin": [(2, 1)]}, [2.0, 5.0], id="skip-to-out2"),
        pytest.param(3, 2, {"inin": [(1, 0), (2, 1)], "outin": [(1, 2)]}, [2.0, 3.0, 5.0], id="two-reused-one-pred"),
    ],
)
def test_copy_prognostic_fields_irregular_advance_map(
    multi_step_input: int,
    multi_step_output: int,
    advance_map: dict,
    expected_prog: list[float],
) -> None:
    """A custom (offset-based) advance map is applied faithfully to the prognostic slots."""
    handler = _make_handler(multi_step_input, multi_step_output, advance_map)
    input_tensor, y_pred = _build_tensors(multi_step_input, multi_step_output)
    check = np.zeros(2, dtype=bool)

    result = handler.copy_prognostic_fields_to_input_tensor(input_tensor, y_pred, check)

    for step, value in enumerate(expected_prog):
        assert torch.all(
            result[0, step, :, 1] == value
        ), f"prognostic step {step} should be {value}, got {result[0, step, :, 1].tolist()}"
    assert check.tolist() == [False, True]


def test_copy_prognostic_fields_preserves_spatial_and_temporal_order() -> None:
    """Per-grid-point prognostic values keep their spatial position while the window advances."""
    handler = _make_handler(2, 1, _regular_advance_map(2, 1))

    input_tensor = torch.tensor(
        [
            [
                [[1.0, 10.0], [1.0, 11.0]],
                [[2.0, 20.0], [2.0, 21.0]],
            ]
        ]
    )
    y_pred = torch.tensor([[[[30.0, -30.0], [31.0, -31.0]]]])
    check = np.zeros(2, dtype=bool)

    result = handler.copy_prognostic_fields_to_input_tensor(input_tensor, y_pred, check)

    # New step 0 keeps the previous step 1 prognostics; new step 1 takes the prediction.
    assert result[0, 0, :, 1].tolist() == [20.0, 21.0]
    assert result[0, 1, :, 1].tolist() == [30.0, 31.0]
    assert check.tolist() == [False, True]
    assert handler._input_kinds["prog"].attributes == {"prognostic": True}


def test_copy_prognostic_fields_raises_on_conflicting_slot() -> None:
    """Writing a prognostic slot that is already marked as filled raises an error."""
    handler = _make_handler(1, 1, _regular_advance_map(1, 1))
    input_tensor, y_pred = _build_tensors(1, 1)
    check = np.array([False, True])  # prognostic input slot already written

    with pytest.raises(AssertionError, match="overwrite existing prognostic input slots"):
        handler.copy_prognostic_fields_to_input_tensor(input_tensor, y_pred, check)
