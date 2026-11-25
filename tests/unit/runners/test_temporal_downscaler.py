# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from types import SimpleNamespace

import pytest

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.inputs.dummy import DummyInput
from anemoi.inference.runners import create_runner
from anemoi.inference.runners.temporal_downscaler import TemporalDownscalerMultiOutRunner
from anemoi.inference.testing import fake_checkpoints
from anemoi.inference.testing.mock_model import SimpleMockModel
from anemoi.inference.testing.mock_model import VariableOutputSetting
from anemoi.inference.types import State


@pytest.mark.parametrize(
    "target_explicit_times",
    [
        pytest.param([1, 2, 3, 4, 5, 6], id="right-boundary"),
        pytest.param([1, 2, 3, 4, 5], id="no-boundary"),
        pytest.param([0, 1, 2, 3, 4, 5], id="left-boundary"),
    ],
)
def test_temporal_downscaler_stepper_target_steps(target_explicit_times: list[int]) -> None:
    runner = object.__new__(TemporalDownscalerMultiOutRunner)
    runner._checkpoint = SimpleNamespace(input_explicit_times=[0, 6], target_explicit_times=target_explicit_times)
    runner.temporal_downscaling_window = datetime.timedelta(hours=6)
    start = datetime.datetime(2017, 1, 9, 6)

    expected_steps = [datetime.timedelta(hours=t) for t in target_explicit_times]
    assert list(runner.temporal_downscaler_stepper(start)) == [(step, start + step) for step in expected_steps]


@pytest.mark.parametrize(
    "write_initial_state, fill_timeline_with_input, expected_hours",
    [
        pytest.param(False, False, [6, 12, 24, 30], id="no-initial-state"),
        pytest.param(True, False, [0, 6, 12, 24, 30], id="initial-state"),
        pytest.param(False, True, [6, 12, 18, 24, 30], id="fill-no-initial-state"),
        pytest.param(True, True, [0, 6, 12, 18, 24, 30], id="fill-initial-state"),
    ],
)
@fake_checkpoints
def test_temporal_downscaler_execute_multiple_windows(
    monkeypatch: pytest.MonkeyPatch,
    write_initial_state: bool,
    fill_timeline_with_input: bool,
    expected_hours: list[int],
) -> None:
    # 18h window of 6h data with no boundary predicted, run over two windows
    config = RunConfiguration(
        checkpoint="unit/checkpoints/multi-single.ckpt",
        date="2020-01-01",
        lead_time="36h",
        device="cpu",
        input="dummy",
        output="none",
        write_initial_state=write_initial_state,
        runner={"temporal_downscaler": {"fill_timeline_with_input": fill_timeline_with_input}},
        patch_metadata={
            "config": {"training": {"multistep_output": 2}},
            "metadata_inference": {
                "task": "temporal-downscaler",
                "data": {
                    "timesteps": {
                        "input_relative_date_indices": [0, 3],
                        "output_relative_date_indices": [1, 2],
                    }
                },
            },
        },
    )
    # Dummy fields are analyses, so they cannot be selected as forecast steps from one reference date
    create_input_state = DummyInput.create_input_state

    def create_analysis_input_state(self: DummyInput, *, select_reference_date: bool = False, **kwargs) -> State:
        return create_input_state(self, **kwargs)

    monkeypatch.setattr(DummyInput, "create_input_state", create_analysis_input_state)

    runner = create_runner(config)
    # The fake checkpoint's model is built from the unpatched metadata, so has a single output step
    runner.model = SimpleMockModel(runner.checkpoint.multi_dataset_metadata, output_mode=VariableOutputSetting.ONES)

    # the task of the metadata looks ahead: inputs at the window start and end, outputs in between
    assert runner.checkpoint.task == "temporal-downscaler"
    assert runner.checkpoint.lagged == [datetime.timedelta(0), datetime.timedelta(hours=18)]
    assert runner.checkpoint.output_offsets == [datetime.timedelta(hours=6), datetime.timedelta(hours=12)]

    written: list[datetime.timedelta] = []

    def record(state: State) -> None:
        written.append(state["step"])

    for output in runner.outputs.values():
        monkeypatch.setattr(output, "write_step", record)

    runner.execute()

    assert written == [datetime.timedelta(hours=h) for h in expected_hours]
