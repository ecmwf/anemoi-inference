# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the parallel runner ``torch.distributed`` process-group teardown.

The process group is torn down once in ``execute`` (not per ``run``), so a
multi-window runner (e.g. the temporal downscaler, which calls ``run`` once per
window) keeps the group alive across all windows. Teardown is idempotent.

See https://github.com/ecmwf/anemoi-inference/issues/570.
"""

from unittest.mock import MagicMock

import pytest

from anemoi.inference.runner import Runner
from anemoi.inference.runners.parallel import ParallelRunnerMixin


class BaseRunnerStub(Runner):
    """A minimal stand-in for a real base runner (e.g. the temporal downscaler).

    It shares the same ``Runner`` ancestry so that mixing it with
    ``ParallelRunnerMixin`` reproduces the real method-resolution order, but it
    skips ``Runner.__init__`` (which requires a checkpoint and config).
    """

    def __init__(self):
        pass


class TestDestroyProcessGroup:
    """`_destroy_process_group` must be a safe, idempotent no-op-if-uninitialised."""

    def test_destroys_when_initialised(self, monkeypatch):
        dist = MagicMock()
        dist.is_available.return_value = True
        dist.is_initialized.return_value = True
        monkeypatch.setattr("anemoi.inference.runners.parallel.torch.distributed", dist)

        ParallelRunnerMixin._destroy_process_group()

        dist.destroy_process_group.assert_called_once()

    def test_noop_when_not_initialised(self, monkeypatch):
        dist = MagicMock()
        dist.is_available.return_value = True
        dist.is_initialized.return_value = False
        monkeypatch.setattr("anemoi.inference.runners.parallel.torch.distributed", dist)

        ParallelRunnerMixin._destroy_process_group()

        dist.destroy_process_group.assert_not_called()

    def test_noop_when_distributed_unavailable(self, monkeypatch):
        dist = MagicMock()
        dist.is_available.return_value = False
        monkeypatch.setattr("anemoi.inference.runners.parallel.torch.distributed", dist)

        ParallelRunnerMixin._destroy_process_group()

        dist.destroy_process_group.assert_not_called()

    def test_idempotent_across_repeated_calls(self, monkeypatch):
        """Calling twice must not raise even after the group is gone."""
        dist = MagicMock()
        dist.is_available.return_value = True
        # First call: initialised; after destroy the group is gone.
        dist.is_initialized.side_effect = [True, False]
        monkeypatch.setattr("anemoi.inference.runners.parallel.torch.distributed", dist)

        ParallelRunnerMixin._destroy_process_group()
        ParallelRunnerMixin._destroy_process_group()

        assert dist.destroy_process_group.call_count == 1


def _make_parallel_runner(base_execute):
    """Build a ParallelRunner over a stub base runner with a custom ``execute``.

    Mirrors the real ``ParallelRunnerFactory.get_class`` mixing: the resulting
    class has MRO ``ParallelRunner -> ParallelRunnerMixin -> BaseRunner -> Runner``,
    so ``ParallelRunnerMixin.execute`` calls ``BaseRunner.execute`` via ``super()``.
    """

    class BaseRunner(BaseRunnerStub):
        pass

    BaseRunner.execute = base_execute
    parallel_cls = type("ParallelRunner", (ParallelRunnerMixin, BaseRunner), {})
    return parallel_cls.__new__(parallel_cls)


class TestExecuteTearsDownOnce:
    """`execute` tears the group down exactly once, after all forecasting."""

    def test_execute_destroys_group_after_super(self, monkeypatch):
        order = []

        def base_execute(self, *args, **kwargs):
            # emulate a multi-window runner (e.g. the temporal downscaler)
            order.append("run-window-1")
            order.append("run-window-2")

        runner = _make_parallel_runner(base_execute)
        monkeypatch.setattr(
            type(runner),
            "_destroy_process_group",
            staticmethod(lambda: order.append("destroy")),
        )

        runner.execute()

        assert order == ["run-window-1", "run-window-2", "destroy"]

    def test_execute_destroys_group_even_on_error(self, monkeypatch):
        destroyed = []

        def base_execute(self, *args, **kwargs):
            raise RuntimeError("boom")

        runner = _make_parallel_runner(base_execute)
        monkeypatch.setattr(
            type(runner),
            "_destroy_process_group",
            staticmethod(lambda: destroyed.append(True)),
        )

        with pytest.raises(RuntimeError, match="boom"):
            runner.execute()

        assert destroyed == [True]
