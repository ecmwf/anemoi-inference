# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The payload a richer-batch checkpoint's ``predict_step`` receives from the runner, and what it returns."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anemoi.inference.runner import RICHER_BATCH_LAYOUT
from anemoi.inference.runner import Runner

N_TIME_IN, N_TIME_OUT, N_GRID = 2, 1, 5
INPUTS = ["2t", "msl", "lsm"]
OUTPUTS = ["2t", "msl", "tp"]


class LayoutModel:
    """Stands in for a richer-batch model: answers in the layout the target template asks for."""

    def __init__(self, returned_layout: tuple[str, ...] | None = None) -> None:
        self.returned_layout = returned_layout
        self.calls = []

    def predict_step(self, x: dict, target_template: dict, **kwargs) -> dict:
        self.calls.append((x, target_template, kwargs))
        out = {}
        for name, template in target_template.items():
            layout = self.returned_layout or tuple(template["layout"])
            sizes = {"time": N_TIME_OUT, "ensemble": 1, "grid": N_GRID, "variables": len(template["variables"])}
            values = torch.arange(float(np.prod([sizes[a] for a in layout]))).reshape(*(sizes[a] for a in layout))
            out[name] = {"values": values, "layout": layout, "variables": template["variables"]}
        return out


def _runner() -> Runner:
    runner = Runner.__new__(Runner)
    runner._checkpoint = SimpleNamespace(multi_dataset=True, richer_batch=True)
    runner.config = SimpleNamespace(predict_kwargs={})
    metadata = SimpleNamespace(
        data_type="gridded",
        variable_to_input_tensor_index={name: i for i, name in enumerate(INPUTS)},
        variable_to_output_tensor_index={name: i for i, name in enumerate(OUTPUTS)},
    )
    runner.tensor_handlers = {"data": SimpleNamespace(metadata=metadata)}
    return runner


def _inputs() -> tuple[dict, dict]:
    tensors = {"data": torch.randn(1, N_TIME_IN, N_GRID, len(INPUTS))}  # (batch, time, grid, variables)
    states = {"data": {"latitudes": np.linspace(-80, 80, N_GRID), "longitudes": np.linspace(0, 300, N_GRID)}}
    return tensors, states


def test_payload_names_the_axes_it_has():
    """No hand-inserted ensemble axis: the payload is (time, grid, variables) and says so."""
    model = LayoutModel()
    tensors, states = _inputs()

    _runner().predict_step(model, tensors, input_states=states)

    ((x, template, _),) = model.calls
    assert x["data"]["layout"] == RICHER_BATCH_LAYOUT == ("time", "grid", "variables")
    assert torch.equal(x["data"]["values"], tensors["data"][0])
    assert x["data"]["variables"] == INPUTS and x["data"]["data_type"] == "gridded"
    assert template["data"]["layout"] == RICHER_BATCH_LAYOUT
    assert template["data"]["variables"] == OUTPUTS
    assert "values" not in template["data"]


def test_prediction_is_returned_with_the_runner_dimensions():
    model = LayoutModel()
    tensors, states = _inputs()

    out = _runner().predict_step(model, tensors, input_states=states)

    # (batch=1, time, ensemble=1, grid, variables), as the forecast loop expects
    assert out["data"].shape == (1, N_TIME_OUT, 1, N_GRID, len(OUTPUTS))


def test_a_prediction_in_another_layout_is_rejected():
    model = LayoutModel(returned_layout=("time", "ensemble", "grid", "variables"))
    tensors, states = _inputs()

    with pytest.raises(ValueError, match="Expected the prediction in the requested layout"):
        _runner().predict_step(model, tensors, input_states=states)
