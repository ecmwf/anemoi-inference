# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from anemoi.metadata import Metadata as CoreMetadata

from anemoi.inference.metadata import InferenceMetadata
from anemoi.inference.tensors import TensorHandler
from anemoi.inference.testing.mock_checkpoint import mock_load_metadata

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
    data = SimpleNamespace(
        prognostic_input_mask=lambda: np.array(prognostic_input_mask),
        prognostic_output_mask=lambda: np.array(prognostic_output_mask),
    )
    timing = SimpleNamespace(n_step_input=multi_step_input, n_step_output=multi_step_output, advance_map=advance_map)
    metadata = SimpleNamespace(
        dataset_name="data",
        core=SimpleNamespace(task=SimpleNamespace(for_dataset=lambda name: timing)),
        data=data,
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
        # Irregular grids that a regular time-shift cannot represent (see anemoi-core ADR-003).
        # The new input window is gathered from a non-uniform mix of previous inputs and
        # predictions, and some previous-input slots may be dropped and never reused.
        # ADR "different input/output frequencies" example ([-6H,0H]->[3H,6H], S=6H).
        pytest.param(2, 2, {"inin": [(1, 0)], "outin": [(1, 1)]}, [2.0, 4.0], id="mixed-frequency"),
        # Tail input slot filled from a non-adjacent, later prediction ([-3H,0H]->[1H,2H,3H], S=3H).
        pytest.param(2, 3, {"inin": [(1, 0)], "outin": [(2, 1)]}, [2.0, 5.0], id="skip-to-out2"),
        # Two previous inputs reused, one prediction fed back ([-4H,-2H,0H]->[1H,2H], S=2H).
        pytest.param(3, 2, {"inin": [(1, 0), (2, 1)], "outin": [(1, 2)]}, [2.0, 3.0, 5.0], id="two-reused-one-pred"),
        # Most irregular case: multiple inin *and* multiple outin, with previous-input
        # slots 0 and 2 dropped ([-4H,-2H,-1H,0H]->[1H,2H], S=2H) — a plain roll() cannot express this.
        pytest.param(
            4, 2, {"inin": [(1, 0), (3, 1)], "outin": [(0, 2), (1, 3)]}, [2.0, 4.0, 5.0, 6.0], id="irregular-offsets"
        ),
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
        assert torch.all(result[0, step, :, 1] == value), (
            f"prognostic step {step} should be {value}, got {result[0, step, :, 1].tolist()}"
        )
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


def test_copy_prognostic_fields_input_only_dataset() -> None:
    """Input-only dataset (no prediction): the window advances, the last step keeps its values."""
    handler = _make_handler(2, 1, _regular_advance_map(2, 1))
    input_tensor, _ = _build_tensors(2, 1)
    check = np.zeros(2, dtype=bool)

    result = handler.copy_prognostic_fields_to_input_tensor(input_tensor, None, check)

    assert result[0, :, :, 1].tolist() == [[2.0, 2.0], [2.0, 2.0]]
    assert check.tolist() == [False, True]


def _real_handler(arrays: dict[str, np.ndarray]) -> TensorHandler:
    """A handler on a real ``InferenceMetadata`` (legacy atmos checkpoint) with the given supporting arrays."""
    raw = mock_load_metadata("unit/checkpoints/atmos.json", supporting_arrays=False)
    metadata = InferenceMetadata(CoreMetadata.from_dict(raw, arrays))
    handler = TensorHandler.__new__(TensorHandler)
    handler.context = SimpleNamespace(allow_nans=False)
    handler.metadata = metadata
    handler.trace = None
    handler.constant_forcings_providers = []
    handler.dynamic_forcings_providers = []
    handler._input_kinds = {}
    handler._input_tensor_by_name = []
    handler._input_units = {}
    return handler


def _n_step_input(handler: TensorHandler) -> int:
    return handler.metadata.core.task.for_dataset(handler.dataset_name).n_step_input


def _input_fields(handler: TensorHandler, n_grid: int) -> dict[str, np.ndarray]:
    steps = _n_step_input(handler)
    return {name: np.zeros((steps, n_grid)) for name in handler._data.variable_indices()}


def test_prepare_input_tensor_defaults_coordinates_to_the_supporting_arrays() -> None:
    """Without coordinates in the state, the latitudes and longitudes of the checkpoint are used."""
    handler = _real_handler({"latitudes": np.array([10.0, 20.0, 30.0]), "longitudes": np.array([0.0, 1.0, 2.0])})
    state = {"date": datetime(2020, 1, 1), "fields": _input_fields(handler, 3)}

    tensor = handler.prepare_input_tensor(state)

    assert state["latitudes"].tolist() == [10.0, 20.0, 30.0]
    assert state["longitudes"].tolist() == [0.0, 1.0, 2.0]
    assert tensor.shape == (_n_step_input(handler), len(handler._data.variable_indices()), 3)


def test_prepare_input_tensor_applies_grid_indices_to_the_coordinates() -> None:
    """The grid points kept by ``grid_indices`` are the ones of the model."""
    handler = _real_handler(
        {
            "latitudes": np.array([10.0, 20.0, 30.0, 40.0]),
            "longitudes": np.array([0.0, 1.0, 2.0, 3.0]),
            "grid_indices": np.array([3, 1]),
        }
    )
    state = {"date": datetime(2020, 1, 1), "fields": _input_fields(handler, 2)}

    tensor = handler.prepare_input_tensor(state)

    assert state["latitudes"].tolist() == [40.0, 20.0]
    assert state["longitudes"].tolist() == [3.0, 1.0]
    assert tensor.shape[-1] == 2


def test_prepare_input_tensor_rejects_coordinates_that_differ_from_the_checkpoint() -> None:
    handler = _real_handler({"latitudes": np.array([10.0, 20.0]), "longitudes": np.array([0.0, 1.0])})
    state = {
        "date": datetime(2020, 1, 1),
        "latitudes": np.array([20.0, 10.0]),
        "longitudes": np.array([0.0, 1.0]),
        "fields": _input_fields(handler, 2),
    }

    with pytest.raises(ValueError, match="Latitudes in the input state do not match metadata"):
        handler.prepare_input_tensor(state)


def test_prepare_input_tensor_reports_missing_variables() -> None:
    handler = _real_handler({"latitudes": np.array([10.0, 20.0]), "longitudes": np.array([0.0, 1.0])})
    fields = _input_fields(handler, 2)
    fields.pop("2t")
    state = {"date": datetime(2020, 1, 1), "fields": fields}

    with pytest.raises(ValueError, match=r"Missing variables in input fields: \['2t'\]"):
        handler.prepare_input_tensor(state)


@pytest.mark.parametrize("with_mask", [False, True])
def test_boundary_forcings_providers_need_an_output_mask(with_mask: bool) -> None:
    """Without an ``output_mask`` supporting array there are no boundary forcings."""
    arrays = {"latitudes": np.array([10.0, 20.0]), "longitudes": np.array([0.0, 1.0])}
    if with_mask:
        arrays["output_mask"] = np.array([True, False])
    handler = _real_handler(arrays)
    created = []
    handler.create_boundary_forcings = lambda variables, mask: created.append(variables) or ["provider"]

    providers = handler.create_boundary_forcings_providers()

    assert providers == (["provider"] if with_mask else [])
    assert bool(created) is with_mask
