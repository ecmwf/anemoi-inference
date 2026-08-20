# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

from anemoi.inference.runner import Runner
from anemoi.inference.runners.parallel import ParallelRunnerMixin


class FakeProcessGroup:
    def __init__(self, rank: int):
        self._rank = rank

    def rank(self) -> int:
        return self._rank


def make_runner(rank: int = 0, is_master: bool = True) -> ParallelRunnerMixin:
    runner = ParallelRunnerMixin.__new__(ParallelRunnerMixin)
    runner.compute_client = SimpleNamespace(process_group=FakeProcessGroup(rank))
    runner.is_master = is_master
    runner.grid_shard_sizes = {"data": [4, 3, 3]}
    return runner


def test_grid_shard_slice_uses_process_group_rank() -> None:
    runner = make_runner(rank=1)

    assert runner.grid_shard_slice("data") == slice(4, 7)
    assert runner.grid_shard_slice("unknown") == slice(None)


def test_predict_step_keeps_input_and_output_sharded(monkeypatch) -> None:
    runner = make_runner(rank=1, is_master=False)
    captured = {}

    def fake_predict_step(self, model, input_tensor, **kwargs):
        captured.update(kwargs)
        return {"data": input_tensor["data"]}

    monkeypatch.setattr(Runner, "predict_step", fake_predict_step)
    input_tensor = {"data": object()}

    result = runner.predict_step(object(), input_tensor)

    assert result == input_tensor
    assert captured["model_comm_group"] is runner.compute_client.process_group
    assert captured["grid_shard_sizes"] == runner.grid_shard_sizes
    assert captured["gather_out"] is False


def test_only_master_writes_output(monkeypatch) -> None:
    writes = []

    def fake_write_output_state(self, dataset, state):
        writes.append((dataset, state))

    monkeypatch.setattr(Runner, "write_output_state", fake_write_output_state)
    state = {"fields": {}}

    make_runner(is_master=False).write_output_state("data", state)
    make_runner(is_master=True).write_output_state("data", state)

    assert writes == [("data", state)]
