# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.processor import Processor
from anemoi.inference.runners import create_runner
from anemoi.inference.testing import fake_checkpoints
from anemoi.inference.testing import files_for_tests


class TestingProcessor(Processor):
    def __init__(self, context, mark: str):
        super().__init__(context)
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
        overrides=dict(runner="testing", device="cpu", input="dummy", trace_path="trace.log"),
    )
    return create_runner(config)


@fake_checkpoints
def test_patched_by_input_and_context(runner):
    runner.pre_processors.append(TestingProcessor(runner, "context"))

    input = runner.create_input()
    input.pre_processors.append(TestingProcessor(runner, "input"))

    empty_request = {}
    patched_request = input.patch_data_request(empty_request)

    assert patched_request["context"] is True
    assert patched_request["input"] is True
