# (C) Copyright 2026- Anemoi contributors.
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

from anemoi.inference.runner import Runner

N_TIME_IN, N_TIME_OUT, N_GRID = 2, 1, 5
INPUTS = ["2t", "msl", "lsm"]
OUTPUTS = ["2t", "msl", "tp"]

GRIDDED_SAMPLE_LAYOUT = ("time", "ensemble", "grid", "variables")


class Model:
    def __init__(self, returned_layout: tuple[str, ...] | None = None) -> None:
        self.returned_layout = returned_layout  # only used by LayoutModel
        self.calls = []


class LayoutModel(Model):
    """Stands in for a richer-batch model: answers in the layout the target template asks for."""

    def predict_step(self, x: dict, target_template: dict, **kwargs) -> dict:
        self.calls.append((x, target_template, kwargs))
        out = {}
        for name, template in target_template.items():
            layout = self.returned_layout or tuple(template["layout"])
            sizes = {"time": N_TIME_OUT, "ensemble": 1, "grid": N_GRID, "variables": len(template["variables"])}
            values = torch.arange(float(np.prod([sizes[a] for a in layout]))).reshape(*(sizes[a] for a in layout))
            out[name] = {"values": values, "layout": layout, "variables": template["variables"]}
        return out


class LegacyModel(Model):
    """Single tensor model. Single output step."""

    def predict_step(self, x: "torch.Tensor", **kwargs) -> "torch.Tensor":
        self.calls.append(x)

        # (batch=1, ensemble=1, values, variables)
        shape = (1, 1, N_GRID, len(OUTPUTS))

        return torch.arange(float(np.prod(shape))).reshape(*shape)


class MultiDatasetModel(Model):
    """Multi-dataset model. Multiple output steps."""

    def predict_step(self, x: dict[str, "torch.Tensor"], **kwargs) -> dict[str, "torch.Tensor"]:
        self.calls.append(x)

        # (batch=1, time, ensemble=1, values, variables)
        shape = (1, N_TIME_OUT, 1, N_GRID, len(OUTPUTS))

        return {dataset: torch.arange(float(np.prod(shape))).reshape(*shape) for dataset in x}


def _runner(multi_dataset: bool, richer_batch: bool) -> Runner:
    runner = Runner.__new__(Runner)
    runner._checkpoint = SimpleNamespace(multi_dataset=multi_dataset, richer_batch=richer_batch)
    runner.config = SimpleNamespace(predict_kwargs={})
    # what `anemoi.metadata` gives for a dataset: `metadata.data`
    data = SimpleNamespace(
        data_type=lambda: "gridded",
        input_index_to_variable=lambda: dict(enumerate(INPUTS)),
        output_index_to_variable=lambda: dict(enumerate(OUTPUTS)),
    )
    metadata = SimpleNamespace(data=data)
    runner.tensor_handlers = {"data": SimpleNamespace(metadata=metadata)}
    return runner


def _inputs() -> tuple[dict, dict]:
    input = {
        "data": torch.randn(1, N_TIME_IN, N_GRID, len(INPUTS))
    }  # input to runner.predict_step (batch, time, grid, variables)
    states = {"data": {"latitudes": np.linspace(-80, 80, N_GRID), "longitudes": np.linspace(0, 300, N_GRID)}}
    return input, states


def test_layout_model():
    model = LayoutModel()
    input, states = _inputs()
    kwargs = {"some": "option", "another": "something"}

    out = _runner(multi_dataset=True, richer_batch=True).predict_step(
        model,
        input,
        input_states=states,
        **kwargs.copy(),
    )

    # tensor returned to the runner must be (batch=1, time, ensemble=1, grid, variables)
    assert out["data"].shape == (1, N_TIME_OUT, 1, N_GRID, len(OUTPUTS))

    ((x, template, model_kwargs),) = model.calls
    assert model_kwargs == kwargs  # kwargs forwarded to the model
    assert torch.equal(
        x["data"]["values"], input["data"][0].unsqueeze(1)
    )  # input tensor passed to the model has batch dimension removed, and ensemble dimension added

    # check that the model receives a correct payload
    assert x["data"]["layout"] == GRIDDED_SAMPLE_LAYOUT
    assert x["data"]["variables"] == INPUTS and x["data"]["data_type"] == "gridded"
    assert template["data"]["layout"] == GRIDDED_SAMPLE_LAYOUT
    assert template["data"]["variables"] == OUTPUTS
    assert "latitudes" in template["data"]
    assert "longitudes" in template["data"]
    assert "data_type" in template["data"]
    assert "values" not in template["data"]


def test_a_prediction_in_another_layout_is_rejected():
    model = LayoutModel(returned_layout=("time", "grid", "variables"))
    input, states = _inputs()

    with pytest.raises(AssertionError, match="Expected output tensor"):
        _runner(multi_dataset=True, richer_batch=True).predict_step(model, input, input_states=states)


def test_legacy_model():
    model = LegacyModel()
    input, _ = _inputs()

    out = _runner(multi_dataset=False, richer_batch=False).predict_step(model, input)

    # tensor returned to the runner must be (batch=1, ensemble=1, grid, variables)
    expected_output_shape = (1, 1, N_GRID, len(OUTPUTS))

    assert out["data"].shape == expected_output_shape
    assert torch.equal(out["data"], torch.arange(float(np.prod(expected_output_shape))).reshape(*expected_output_shape))

    (x,) = model.calls
    assert torch.equal(x, input["data"])  # input tensor passed to the model as is


def test_multi_dataset_model():
    model = MultiDatasetModel()
    input, _ = _inputs()
    input["more_data"] = torch.clone(input["data"])

    out = _runner(multi_dataset=True, richer_batch=False).predict_step(model, input)

    # tensor returned to the runner must be (batch=1, time, ensemble=1, grid, variables)
    expected_output_shape = (1, N_TIME_OUT, 1, N_GRID, len(OUTPUTS))

    (x,) = model.calls
    for key in ["data", "more_data"]:
        assert out[key].shape == expected_output_shape
        assert torch.equal(
            out[key], torch.arange(float(np.prod(expected_output_shape))).reshape(*expected_output_shape)
        )
        assert torch.equal(x[key], input[key])  # input tensor passed to the model as is
