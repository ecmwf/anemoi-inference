# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.inputs.dummy import DummyInput
from anemoi.inference.inputs.ekd import RequestInput
from anemoi.inference.processor import Processor
from anemoi.inference.runners import create_runner
from anemoi.inference.testing import fake_checkpoints
from anemoi.inference.testing import files_for_tests


class DummyProcessor(Processor):
    def __init__(self, context, metadata, mark: str):
        super().__init__(context, metadata)
        self.mark = mark

    def process(self, data: dict) -> dict:  # type: ignore
        """A simple processor that returns the input data unchanged."""
        return data

    def patch_data_request(self, data: dict) -> dict:  # type: ignore
        """A simple patch method that returns the input data unchanged."""
        data[self.mark] = True
        return data


@pytest.fixture
@fake_checkpoints
def runner() -> None:
    config = RunConfiguration.load(
        files_for_tests("unit/configs/simple.yaml"),
        overrides=dict(runner="default", device="cpu", input="dummy"),
    )
    return create_runner(config)


@fake_checkpoints
def test_patched_by_input_and_context(runner):
    metadata = runner.checkpoint._metadata
    runner.pre_processors["data"].append(DummyProcessor(runner, metadata, "context"))

    input = DummyInput(runner, metadata)
    input.pre_processors.append(DummyProcessor(runner, metadata, "input"))

    empty_request = {}
    patched_request = input.patch_data_request(empty_request)

    assert patched_request["context"] is True
    assert patched_request["input"] is True


class _RequestInput(RequestInput):
    """Minimal concrete RequestInput for exercising request patching."""

    trace_name = "test-request"

    def _retrieve(self, requests, **kwargs):  # pragma: no cover - not used here
        raise NotImplementedError


def make_request_input(*, from_forecast, reference_date):
    """Build a `RequestInput` with a mocked context/metadata.

    The context/pre-processors are neutralised so that
    `patch_data_request` only exercises the `from_forecast` logic.
    """
    context = MagicMock()
    context.reference_date = reference_date
    # Context passes the request through unchanged.
    context.patch_data_request.side_effect = lambda request, dataset_name: request

    metadata = MagicMock()
    metadata.dataset_name = "test"
    metadata.default_namer.return_value = lambda field, original_metadata: "var"

    input = _RequestInput(
        context,
        metadata,
        variables=["2t"],
        from_forecast=from_forecast,
    )
    # Neutralise pre-processors so patch_data_request is a no-op there.
    input.__dict__["pre_processors"] = []
    return input


def test_from_forecast_false_leaves_dates_untouched():
    input = make_request_input(from_forecast=False, reference_date=datetime(2024, 6, 1, 0))

    request = {"date": ["2024-06-01"], "time": ["0600"], "param": ["2t"]}
    patched = input.patch_data_request(dict(request))

    assert patched["date"] == ["2024-06-01"]
    assert patched["time"] == ["0600"]
    assert "step" not in patched


def test_from_forecast_converts_dates_to_steps():
    base_date = datetime(2024, 6, 1, 0)
    input = make_request_input(from_forecast=True, reference_date=base_date)

    request = {"date": ["2024-06-01"], "time": ["0000", "0600"], "param": ["2t"]}
    patched = input.patch_data_request(dict(request))

    assert patched["step"] == [0, 6]
    assert patched["date"] == ["2024-06-01"]
    assert patched["time"] == ["0000"]


def test_from_forecast_uses_reference_date_not_request_date():
    base_date = datetime(2024, 6, 1, 6)
    input = make_request_input(from_forecast=True, reference_date=base_date)

    # Requested valid dates span two days; steps are relative to the base date.
    request = {"date": ["2024-06-01", "2024-06-02"], "time": ["1200"], "param": ["2t"]}
    patched = input.patch_data_request(dict(request))

    assert patched["step"] == [6, 30]
    assert patched["date"] == ["2024-06-01"]
    assert patched["time"] == ["0600"]


def test_from_forecast_requires_reference_date():
    input = make_request_input(from_forecast=True, reference_date=None)

    request = {"date": ["2024-06-01"], "time": ["0000"], "param": ["2t"]}
    with pytest.raises(ValueError, match="Reference date is not set"):
        input.patch_data_request(dict(request))
