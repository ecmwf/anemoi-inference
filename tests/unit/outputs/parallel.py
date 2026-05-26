# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from anemoi.inference.decorators import supports_parallel_output
from anemoi.inference.outputs.parallel import MessageType
from anemoi.inference.outputs.parallel import _detach_tensors
from anemoi.inference.outputs.parallel import _get_state_chunk
from anemoi.inference.outputs.parallel import _sanitise_state

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_state(n_fields, extra=None):
    state = {"fields": {f"field_{i}": float(i) for i in range(n_fields)}, "date": "2020-01-01"}
    if extra:
        state.update(extra)
    return state


# ── _detach_tensors ───────────────────────────────────────────────────────────


class TestDetachTensors:
    def test_passthrough_when_no_torch(self):
        """Plain Python objects should come out unchanged."""
        obj = {"a": 1, "b": [2, 3]}
        assert _detach_tensors(obj) == obj

    def test_dict_recursed(self):
        d = {"x": 1.0, "nested": {"y": 2.0}}
        result = _detach_tensors(d)
        assert result == d

    def test_list_recursed(self):
        lst = [1, 2, [3, 4]]
        result = _detach_tensors(lst)
        assert result == lst

    def test_tuple_preserved_as_tuple(self):
        t = (1, 2, 3)
        result = _detach_tensors(t)
        assert isinstance(result, tuple)
        assert result == t

    def test_torch_tensor_converted(self):
        pytest.importorskip("torch")
        import torch

        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = _detach_tensors(tensor)
        import numpy as np

        assert isinstance(result, np.ndarray)
        assert list(result) == pytest.approx([1.0, 2.0, 3.0])

    def test_tensor_inside_dict(self):
        pytest.importorskip("torch")
        import numpy as np
        import torch

        state = {"values": torch.tensor([4.0, 5.0])}
        result = _detach_tensors(state)
        assert isinstance(result["values"], np.ndarray)


# ── _sanitise_state ───────────────────────────────────────────────────────────


class TestSanitiseState:
    def test_unpicklable_keys_removed(self):
        state = _make_state(2, {"_grib_templates_for_output": "x", "_input": "y"})
        result = _sanitise_state(state)
        assert "_grib_templates_for_output" not in result
        assert "_input" not in result

    def test_other_keys_preserved(self):
        state = _make_state(2, {"date": "2020-01-01", "step": 6})
        result = _sanitise_state(state)
        assert result["date"] == "2020-01-01"
        assert result["step"] == 6

    def test_original_not_mutated(self):
        state = _make_state(2, {"_grib_templates_for_output": "x"})
        original_keys = set(state.keys())
        _sanitise_state(state)
        assert set(state.keys()) == original_keys

    def test_absent_unpicklable_keys_ignored(self):
        state = _make_state(2)
        result = _sanitise_state(state)
        assert "fields" in result


# ── _get_state_chunk ──────────────────────────────────────────────────────────


class TestGetStateChunk:
    def test_single_chunk_returns_full_state(self):
        state = _make_state(6)
        chunk = _get_state_chunk(state, num_chunks=1, index=0)
        assert chunk is state  # identity, not a copy

    def test_two_chunks_split_evenly(self):
        state = _make_state(6)
        c0 = _get_state_chunk(state, 2, 0)
        c1 = _get_state_chunk(state, 2, 1)
        assert list(c0["fields"].keys()) == ["field_0", "field_1", "field_2"]
        assert list(c1["fields"].keys()) == ["field_3", "field_4", "field_5"]

    def test_chunks_cover_all_fields(self):
        state = _make_state(10)
        all_fields = set()
        for i in range(4):
            chunk = _get_state_chunk(state, 4, i)
            all_fields.update(chunk["fields"].keys())
        assert all_fields == set(state["fields"].keys())

    def test_chunks_are_disjoint(self):
        state = _make_state(12)
        chunks = [_get_state_chunk(state, 4, i) for i in range(4)]
        seen = []
        for chunk in chunks:
            for k in chunk["fields"]:
                assert k not in seen, f"{k} appears in more than one chunk"
                seen.append(k)

    def test_uneven_split(self):
        """ceil division: 7 fields / 3 writers → [3, 3, 1]"""
        state = _make_state(7)
        sizes = [len(_get_state_chunk(state, 3, i)["fields"]) for i in range(3)]
        assert sum(sizes) == 7
        assert sizes[0] == 3
        assert sizes[1] == 3
        assert sizes[2] == 1

    def test_single_field(self):
        state = _make_state(1)
        c0 = _get_state_chunk(state, 2, 0)
        c1 = _get_state_chunk(state, 2, 1)
        assert len(c0["fields"]) == 1
        assert len(c1["fields"]) == 0

    def test_more_writers_than_fields(self):
        """Extra writers get empty chunks, no error."""
        state = _make_state(2)
        sizes = [len(_get_state_chunk(state, 4, i)["fields"]) for i in range(4)]
        assert sum(sizes) == 2

    def test_non_field_keys_preserved_in_all_chunks(self):
        state = _make_state(6, {"step": 12, "date": "2020-01-01"})
        for i in range(3):
            chunk = _get_state_chunk(state, 3, i)
            assert chunk["step"] == 12
            assert chunk["date"] == "2020-01-01"

    def test_chunk_is_shallow_copy_of_state(self):
        state = _make_state(4)
        chunk = _get_state_chunk(state, 2, 0)
        assert chunk is not state
        assert chunk["fields"] is not state["fields"]

    def test_missing_fields_raises(self):
        with pytest.raises(ValueError, match="fields"):
            _get_state_chunk({"date": "2020-01-01"}, 2, 0)

    def test_index_out_of_range_raises(self):
        state = _make_state(4)
        with pytest.raises(AssertionError):
            _get_state_chunk(state, 2, 5)

    def test_negative_index_raises(self):
        state = _make_state(4)
        with pytest.raises(AssertionError):
            _get_state_chunk(state, 2, -1)


# ── ParallelOutput (unit, no subprocesses) ────────────────────────────────────


def _make_parallel_output(num_writers=2):
    """Build a ParallelOutput with mocked context/metadata, no processes spawned."""
    from anemoi.inference.outputs.parallel import ParallelOutput

    context = MagicMock()
    context.reference_date = "2020-01-01"
    context.typed_variables = {}
    context.output_frequency = None

    metadata = MagicMock()
    metadata.dataset_name = "test"
    metadata.typed_variables = {}

    po = ParallelOutput.__new__(ParallelOutput)
    po.context = context
    po.metadata = metadata
    po.num_writers = num_writers
    po.output_config = {"grib": {"path": "out.grib"}}
    po.kwargs = {}
    po._writers_running = True
    po._post_processor_confs = []
    po._write_step_zero = None
    po._output_frequency = None
    po.variables = None
    po.typed_variables = {}
    po.dataset_name = "test"
    po.reference_date = "2020-01-01"
    return po


class TestParallelOutputDispatch:
    def _make_queues_and_processes(self, num_writers, alive=True):
        queues = [MagicMock() for _ in range(num_writers)]
        processes = [MagicMock() for _ in range(num_writers)]
        for p in processes:
            p.is_alive.return_value = alive
        return queues, processes

    def test_dispatch_puts_to_all_queues(self):
        po = _make_parallel_output(num_writers=3)
        po._queues, po._processes = self._make_queues_and_processes(3)

        state = _make_state(6)
        po.dispatch_state_to_writers(state, message=MessageType.STATE)

        for q in po._queues:
            q.put.assert_called_once()

    def test_dispatch_sends_correct_message_type(self):
        po = _get_parallel_output = _make_parallel_output(num_writers=2)
        po._queues, po._processes = self._make_queues_and_processes(2)

        state = _make_state(4)
        po.dispatch_state_to_writers(state, message=MessageType.INITIAL_STATE)

        for q in po._queues:
            args, _ = q.put.call_args
            _payload, msg_type = args[0]
            assert msg_type == MessageType.INITIAL_STATE

    def test_each_writer_gets_disjoint_chunk(self):
        po = _make_parallel_output(num_writers=2)
        po._queues, po._processes = self._make_queues_and_processes(2)

        state = _make_state(4)
        po.dispatch_state_to_writers(state, message=MessageType.STATE)

        received_fields = []
        for q in po._queues:
            args, _ = q.put.call_args
            payload, _ = args[0]
            received_fields.append(set(payload["fields"].keys()))

        assert received_fields[0].isdisjoint(received_fields[1])
        assert received_fields[0] | received_fields[1] == set(state["fields"].keys())

    def test_write_state_calls_dispatch(self):
        po = _make_parallel_output(num_writers=1)
        po._queues, po._processes = self._make_queues_and_processes(1)

        state = _make_state(2)
        po.write_state(state)
        po._queues[0].put.assert_called_once()

    def test_write_initial_state_uses_correct_message_type(self):
        po = _make_parallel_output(num_writers=1)
        po._queues, po._processes = self._make_queues_and_processes(1)

        state = _make_state(2)
        po.write_initial_state(state)

        args, _ = po._queues[0].put.call_args
        _, msg_type = args[0]
        assert msg_type == MessageType.INITIAL_STATE


class TestParallelOutputWriterAliveCheck:
    def test_dead_writer_raises_runtime_error(self):
        po = _make_parallel_output(num_writers=2)
        po._queues = [MagicMock(), MagicMock()]
        po._processes = [MagicMock(), MagicMock()]
        po._processes[0].is_alive.return_value = False
        po._processes[1].is_alive.return_value = True

        with pytest.raises(RuntimeError, match="Writer 0 is dead"):
            po._check_writer_alive(0)

    def test_dead_writer_cancels_all_queues(self):
        po = _make_parallel_output(num_writers=2)
        po._queues = [MagicMock(), MagicMock()]
        po._processes = [MagicMock(), MagicMock()]
        po._processes[0].is_alive.return_value = False

        with pytest.raises(RuntimeError):
            po._check_writer_alive(0)

        for q in po._queues:
            q.cancel_join_thread.assert_called_once()

    def test_alive_writer_does_not_raise(self):
        po = _make_parallel_output(num_writers=1)
        po._queues = [MagicMock()]
        po._processes = [MagicMock()]
        po._processes[0].is_alive.return_value = True
        po._check_writer_alive(0)  # should not raise


class TestParallelOutputInit:
    def _make(self, **kwargs):
        from anemoi.inference.outputs.parallel import ParallelOutput

        context = MagicMock()
        context.reference_date = "2020-01-01"
        context.typed_variables = {}
        context.output_frequency = None

        metadata = MagicMock()
        metadata.dataset_name = "test"
        metadata.typed_variables = {}

        return ParallelOutput(context, metadata, **kwargs)

    def test_num_writers_zero_raises(self):
        with pytest.raises(ValueError, match="num_writers"):
            self._make(num_writers=0, output={"grib": {"path": "out.grib"}})

    def test_num_writers_negative_raises(self):
        with pytest.raises(ValueError, match="num_writers"):
            self._make(num_writers=-1, output={"grib": {"path": "out.grib"}})

    def test_default_num_writers_is_one(self):
        po = self._make(output={"grib": {"path": "out.grib"}})
        assert po.num_writers == 1

    def test_output_config_stored(self):
        cfg = {"grib": {"path": "out.grib"}}
        po = self._make(output=cfg)
        assert po.output_config == cfg

    def test_short_syntax_stores_kwargs_as_output_config(self):
        """When output= is omitted, remaining kwargs become the output config."""
        po = self._make(num_writers=2, grib={"path": "out.grib"})
        assert po.output_config == {"grib": {"path": "out.grib"}}

    def test_writers_not_running_at_init(self):
        po = self._make(output={"grib": {"path": "out.grib"}})
        assert po._writers_running is False

    def test_repr(self):
        po = self._make(num_writers=2, output={"grib": {"path": "out.grib"}})
        assert "ParallelOutput" in repr(po)
        assert "num_writers=2" in repr(po)


# ── supports_parallel_output.add_suffix_to_path ──────────────────────────────


class TestAddSuffixToPath:
    def test_plain_string(self):
        assert supports_parallel_output.add_suffix_to_path("output.grib", "_w0") == "output_w0.grib"

    def test_plain_path_object(self):
        result = supports_parallel_output.add_suffix_to_path(Path("output.grib"), "_w1")
        assert result == "output_w1.grib"

    def test_nested_path(self):
        assert supports_parallel_output.add_suffix_to_path("/some/dir/forecast.nc", "_w2") == "/some/dir/forecast_w2.nc"

    def test_no_extension(self):
        assert supports_parallel_output.add_suffix_to_path("output", "_w0") == "output_w0"

    def test_url_http(self):
        result = supports_parallel_output.add_suffix_to_path("https://example.com/path/forecast.zarr", "_w0")
        assert result == "https://example.com/path/forecast_w0.zarr"

    def test_url_s3(self):
        result = supports_parallel_output.add_suffix_to_path("s3://mybucket/forecasts/out.zarr", "_w3")
        assert result == "s3://mybucket/forecasts/out_w3.zarr"

    def test_url_scheme_preserved(self):
        result = supports_parallel_output.add_suffix_to_path("s3://mybucket/out.zarr", "_w0")
        assert result.startswith("s3://mybucket/")

    def test_non_path_store_returns_unchanged(self):
        store = object()
        result = supports_parallel_output.add_suffix_to_path(store, "_w0")
        assert result is store

    def test_non_path_store_logs_warning(self, caplog):
        import logging

        store = object()
        with caplog.at_level(logging.WARNING):
            supports_parallel_output.add_suffix_to_path(store, "_w0")
        assert "cannot apply suffix" in caplog.text


# ── supports_parallel_output decorator ───────────────────────────────────────


class TestSupportsParallelOutputDecorator:
    def _make_cls(self, arg="path"):
        @supports_parallel_output(arg)
        class _Output:
            def __init__(self, context, metadata, *, path=None, store=None, **kwargs):
                self.path = path
                self.store = store

        return _Output

    def test_suffix_applied_to_path(self):
        cls = self._make_cls("path")
        obj = cls("ctx", "meta", path="out.grib", **{"_parallel-output-suffix": "_w0"})
        assert obj.path == "out_w0.grib"

    def test_no_suffix_leaves_path_unchanged(self):
        cls = self._make_cls("path")
        obj = cls("ctx", "meta", path="out.grib")
        assert obj.path == "out.grib"

    def test_suffix_not_in_kwargs_after_init(self):
        @supports_parallel_output("path")
        class _Output:
            def __init__(self, context, metadata, *, path=None, **kwargs):
                self.extra_kwargs = kwargs

        obj = _Output("ctx", "meta", path="out.grib", **{"_parallel-output-suffix": "_w0"})
        assert "_parallel-output-suffix" not in obj.extra_kwargs

    def test_none_path_not_modified(self):
        cls = self._make_cls("path")
        obj = cls("ctx", "meta", path=None, **{"_parallel-output-suffix": "_w0"})
        assert obj.path is None

    def test_invalid_suffix_type_raises(self):
        cls = self._make_cls("path")
        with pytest.raises(ValueError, match="_parallel-output-suffix"):
            cls("ctx", "meta", path="out.grib", **{"_parallel-output-suffix": 42})

    def test_marks_class_attribute(self):
        @supports_parallel_output("path")
        class _Output:
            def __init__(self, *a, **kw):
                pass

        assert getattr(_Output, "_supports_parallel_output", False)
