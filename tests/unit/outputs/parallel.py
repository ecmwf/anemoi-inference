# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import re
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from anemoi.inference.decorators import supports_parallel_output
from anemoi.inference.outputs.parallel import MessageType
from anemoi.inference.outputs.parallel import ParallelOutput
from anemoi.inference.outputs.parallel import _detach_tensors
from anemoi.inference.outputs.parallel import _get_state_chunk
from anemoi.inference.outputs.parallel import _sanitise_state
from anemoi.inference.outputs.printer import PrinterOutput

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_state(n_fields, extra=None):
    state = {"fields": {f"field_{i}": np.array([float(i)]) for i in range(n_fields)}, "date": "2020-01-01"}
    if extra:
        state.update(extra)
    return state


def _make_context_and_metadata():
    context = MagicMock()
    context.reference_date = "2020-01-01"
    context.typed_variables = {}
    context.output_frequency = None
    context.allow_nans = False

    metadata = MagicMock()
    metadata.dataset_name = "test"
    metadata.typed_variables = {}
    return context, metadata


# ── PrinterOutput helpers for correctness tests ──────────────────────────────
#
# PrinterOutput writes human-readable text lines of the form:
#   field_0   shape=(1,) min=0                  max=0
# We parse those lines to recover (field_name, min, max) records.

_FIELD_LINE_RE = re.compile(r"^\s{4}(\S+)\s+shape=\S+\s+min=(\S+)\s+max=(\S+)")


def _parse_printer_output(path: Path) -> list[dict]:
    """Extract {name, min, max} records from a PrinterOutput text file."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        m = _FIELD_LINE_RE.match(line)
        if m:
            records.append({"name": m.group(1), "min": float(m.group(2)), "max": float(m.group(3))})
    return records


def _collect_all_printer_records(base_path: Path, num_writers: int) -> list[dict]:
    """Read and merge printer records from all per-writer text files."""
    records = []
    for i in range(num_writers):
        p = base_path.with_stem(base_path.stem + f"_w{i}")
        records.extend(_parse_printer_output(p))
    return records


# ── Sequential vs Parallel correctness tests ─────────────────────────────────


def _run_sequential(context, metadata, path, states):
    """Write states through a plain PrinterOutput (no parallelism)."""
    output = PrinterOutput(context, metadata, path=str(path), max_lines=0)
    output.open(states[0])
    for state in states:
        output.write_state(state)
    output.close()
    return _parse_printer_output(path)


def _run_parallel(context, metadata, path, states, num_writers):
    """Write states through ParallelOutput wrapping PrinterOutput."""
    config = {"printer": {"path": str(path), "max_lines": 0}}
    po = ParallelOutput(context, metadata, output=config, num_writers=num_writers)
    po.open(states[0])
    for state in states:
        po.write_state(state)
    po.close()
    return _collect_all_printer_records(path, num_writers)


def _records_equal(seq_records, par_records):
    """Compare two record lists ignoring order (parallel output reorders fields)."""

    def key(r):
        return (r["name"], r["min"], r["max"])

    return sorted(seq_records, key=key) == sorted(par_records, key=key)


class TestParallelOutputCorrectness:
    """End-to-end correctness tests: parallel output must produce the same
    field names and values as sequential output (via PrinterOutput files).
    """

    def test_single_writer_matches_sequential(self, tmp_path):
        context, metadata = _make_context_and_metadata()
        states = [_make_state(6, {"step": datetime.timedelta(hours=h)}) for h in range(3)]

        seq = _run_sequential(context, metadata, tmp_path / "seq.txt", states)
        par = _run_parallel(context, metadata, tmp_path / "par.txt", states, num_writers=1)

        assert _records_equal(seq, par), f"single writer mismatch:\nseq={seq}\npar={par}"

    def test_two_writers_covers_all_fields(self, tmp_path):
        context, metadata = _make_context_and_metadata()
        states = [_make_state(8, {"step": datetime.timedelta(hours=h)}) for h in range(2)]

        seq = _run_sequential(context, metadata, tmp_path / "seq.txt", states)
        par = _run_parallel(context, metadata, tmp_path / "par.txt", states, num_writers=2)

        assert _records_equal(seq, par), "2-writer parallel output differs from sequential"

    def test_four_writers_covers_all_fields(self, tmp_path):
        context, metadata = _make_context_and_metadata()
        states = [_make_state(12, {"step": datetime.timedelta(hours=h)}) for h in range(4)]

        seq = _run_sequential(context, metadata, tmp_path / "seq.txt", states)
        par = _run_parallel(context, metadata, tmp_path / "par.txt", states, num_writers=4)

        assert _records_equal(seq, par), "4-writer parallel output differs from sequential"

    def test_uneven_field_count(self, tmp_path):
        """7 fields / 3 writers: last writer gets fewer fields."""
        context, metadata = _make_context_and_metadata()
        states = [_make_state(7, {"step": datetime.timedelta(hours=h)}) for h in range(2)]

        seq = _run_sequential(context, metadata, tmp_path / "seq.txt", states)
        par = _run_parallel(context, metadata, tmp_path / "par.txt", states, num_writers=3)

        assert _records_equal(seq, par), "uneven split: parallel output differs from sequential"

    def test_no_duplicate_fields_across_writers(self, tmp_path):
        """Each field name must appear exactly once across all writer files per step."""
        context, metadata = _make_context_and_metadata()
        states = [_make_state(6, {"step": datetime.timedelta(hours=h)}) for h in range(3)]

        base = tmp_path / "par.txt"
        _run_parallel(context, metadata, base, states, num_writers=3)
        all_records = _collect_all_printer_records(base, 3)

        # Count occurrences of each field name
        from collections import Counter

        counts = Counter(r["name"] for r in all_records)
        n_steps = len(states)
        for name, count in counts.items():
            assert count == n_steps, f"Field {name!r} appears {count} times, expected {n_steps}"

    def test_field_values_preserved(self, tmp_path):
        """Min/max values written by parallel output must match sequential output."""
        context, metadata = _make_context_and_metadata()
        states = [_make_state(4, {"step": datetime.timedelta(hours=h)}) for h in range(2)]

        seq = _run_sequential(context, metadata, tmp_path / "seq.txt", states)
        par = _run_parallel(context, metadata, tmp_path / "par.txt", states, num_writers=2)

        seq_map = {r["name"]: (r["min"], r["max"]) for r in seq}
        par_map = {r["name"]: (r["min"], r["max"]) for r in par}

        assert seq_map == par_map, "field values differ between sequential and parallel"

    def test_each_writer_produces_its_own_file(self, tmp_path):
        context, metadata = _make_context_and_metadata()
        states = [_make_state(4, {"step": datetime.timedelta(hours=0)})]
        base = tmp_path / "out.txt"

        _run_parallel(context, metadata, base, states, num_writers=3)

        for i in range(3):
            p = base.with_stem(base.stem + f"_w{i}")
            assert p.exists(), f"Writer {i} did not produce a file at {p}"


# ── _detach_tensors ───────────────────────────────────────────────────────────


class TestDetachTensors:
    def test_passthrough_when_no_torch(self):
        obj = {"a": 1, "b": [2, 3]}
        assert _detach_tensors(obj) == obj

    def test_dict_recursed(self):
        d = {"x": 1.0, "nested": {"y": 2.0}}
        assert _detach_tensors(d) == d

    def test_list_recursed(self):
        lst = [1, 2, [3, 4]]
        assert _detach_tensors(lst) == lst

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
        assert chunk is state

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
        state = _make_state(7)
        sizes = [len(_get_state_chunk(state, 3, i)["fields"]) for i in range(3)]
        assert sum(sizes) == 7
        assert sizes == [3, 3, 1]

    def test_single_field(self):
        state = _make_state(1)
        c0 = _get_state_chunk(state, 2, 0)
        c1 = _get_state_chunk(state, 2, 1)
        assert len(c0["fields"]) == 1
        assert len(c1["fields"]) == 0

    def test_more_writers_than_fields(self):
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
    context, metadata = _make_context_and_metadata()
    po = ParallelOutput.__new__(ParallelOutput)
    po.context = context
    po.metadata = metadata
    po.num_writers = num_writers
    po.output_config = {"recording": {"path": "out.json"}}
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
        po.dispatch_state_to_writers(_make_state(6), message=MessageType.STATE)
        for q in po._queues:
            q.put.assert_called_once()

    def test_dispatch_sends_correct_message_type(self):
        po = _make_parallel_output(num_writers=2)
        po._queues, po._processes = self._make_queues_and_processes(2)
        po.dispatch_state_to_writers(_make_state(4), message=MessageType.INITIAL_STATE)
        for q in po._queues:
            args, _ = q.put.call_args
            _, msg_type = args[0]
            assert msg_type == MessageType.INITIAL_STATE

    def test_each_writer_gets_disjoint_chunk(self):
        po = _make_parallel_output(num_writers=2)
        po._queues, po._processes = self._make_queues_and_processes(2)
        state = _make_state(4)
        po.dispatch_state_to_writers(state, message=MessageType.STATE)

        received = []
        for q in po._queues:
            args, _ = q.put.call_args
            payload, _ = args[0]
            received.append(set(payload["fields"].keys()))

        assert received[0].isdisjoint(received[1])
        assert received[0] | received[1] == set(state["fields"].keys())

    def test_write_state_calls_dispatch(self):
        po = _make_parallel_output(num_writers=1)
        po._queues, po._processes = self._make_queues_and_processes(1)
        po.write_state(_make_state(2))
        po._queues[0].put.assert_called_once()

    def test_write_initial_state_uses_correct_message_type(self):
        po = _make_parallel_output(num_writers=1)
        po._queues, po._processes = self._make_queues_and_processes(1)
        po.write_initial_state(_make_state(2))
        args, _ = po._queues[0].put.call_args
        _, msg_type = args[0]
        assert msg_type == MessageType.INITIAL_STATE


class TestParallelOutputWriterAliveCheck:
    def test_dead_writer_raises_runtime_error(self):
        po = _make_parallel_output(num_writers=2)
        po._queues = [MagicMock(), MagicMock()]
        po._processes = [MagicMock(), MagicMock()]
        po._processes[0].is_alive.return_value = False
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
        po._check_writer_alive(0)


class TestParallelOutputInit:
    def _make(self, **kwargs):
        context, metadata = _make_context_and_metadata()
        return ParallelOutput(context, metadata, **kwargs)

    def test_num_writers_zero_raises(self):
        with pytest.raises(ValueError, match="num_writers"):
            self._make(num_writers=0, output={"recording": {"path": "out.json"}})

    def test_num_writers_negative_raises(self):
        with pytest.raises(ValueError, match="num_writers"):
            self._make(num_writers=-1, output={"recording": {"path": "out.json"}})

    def test_default_num_writers_is_one(self):
        po = self._make(output={"recording": {"path": "out.json"}})
        assert po.num_writers == 1

    def test_output_config_stored(self):
        cfg = {"recording": {"path": "out.json"}}
        po = self._make(output=cfg)
        assert po.output_config == cfg

    def test_writers_not_running_at_init(self):
        po = self._make(output={"recording": {"path": "out.json"}})
        assert po._writers_running is False

    def test_repr(self):
        po = self._make(num_writers=2, output={"recording": {"path": "out.json"}})
        assert "ParallelOutput" in repr(po)
        assert "num_writers=2" in repr(po)


# ── supports_parallel_output ──────────────────────────────────────────────────


class TestAddSuffixToPath:
    def test_plain_string(self):
        assert supports_parallel_output.add_suffix_to_path("output.grib", "_w0") == "output_w0.grib"

    def test_plain_path_object(self):
        assert supports_parallel_output.add_suffix_to_path(Path("output.grib"), "_w1") == "output_w1.grib"

    def test_nested_path(self):
        assert supports_parallel_output.add_suffix_to_path("/some/dir/forecast.nc", "_w2") == "/some/dir/forecast_w2.nc"

    def test_no_extension(self):
        assert supports_parallel_output.add_suffix_to_path("output", "_w0") == "output_w0"

    def test_url_s3(self):
        assert supports_parallel_output.add_suffix_to_path("s3://bucket/out.zarr", "_w3") == "s3://bucket/out_w3.zarr"

    def test_url_scheme_preserved(self):
        result = supports_parallel_output.add_suffix_to_path("s3://bucket/out.zarr", "_w0")
        assert result.startswith("s3://bucket/")

    def test_non_path_store_returns_unchanged(self):
        store = object()
        assert supports_parallel_output.add_suffix_to_path(store, "_w0") is store

    def test_non_path_store_logs_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            supports_parallel_output.add_suffix_to_path(object(), "_w0")
        assert "cannot apply suffix" in caplog.text


class TestSupportsParallelOutputDecorator:
    def _make_cls(self):
        @supports_parallel_output("path")
        class _Output:
            def __init__(self, context, metadata, *, path=None, **kwargs):
                self.path = path

        return _Output

    def test_suffix_applied(self):
        obj = self._make_cls()("ctx", "meta", path="out.grib", **{"_parallel-output-suffix": "_w0"})
        assert obj.path == "out_w0.grib"

    def test_no_suffix_unchanged(self):
        obj = self._make_cls()("ctx", "meta", path="out.grib")
        assert obj.path == "out.grib"

    def test_suffix_consumed_not_forwarded(self):
        @supports_parallel_output("path")
        class _O:
            def __init__(self, c, m, *, path=None, **kw):
                self.kw = kw

        obj = _O("c", "m", path="out.grib", **{"_parallel-output-suffix": "_w0"})
        assert "_parallel-output-suffix" not in obj.kw

    def test_none_path_not_modified(self):
        obj = self._make_cls()("ctx", "meta", path=None, **{"_parallel-output-suffix": "_w0"})
        assert obj.path is None

    def test_invalid_suffix_type_raises(self):
        with pytest.raises(ValueError, match="_parallel-output-suffix"):
            self._make_cls()("ctx", "meta", path="out.grib", **{"_parallel-output-suffix": 42})

    def test_marks_class_attribute(self):
        @supports_parallel_output("path")
        class _O:
            def __init__(self, *a, **kw):
                pass

        assert getattr(_O, "_supports_parallel_output", False)

    def _make_nested_cls(self, *paths):
        @supports_parallel_output(*paths)
        class _Output:
            def __init__(self, path=None, archive_requests=None, **kwargs):
                self.path = path
                self.archive_requests = archive_requests

        return _Output

    def test_nested_suffix_applied(self):
        obj = self._make_nested_cls("archive_requests.path")(
            archive_requests={"path": "out.json"}, **{"_parallel-output-suffix": "_w0"}
        )
        assert obj.archive_requests["path"] == "out_w0.json"

    def test_multiple_args_applied(self):
        obj = self._make_nested_cls("path", "archive_requests.path")(
            path="out.grib", archive_requests={"path": "out.json"}, **{"_parallel-output-suffix": "_w0"}
        )
        assert obj.path == "out_w0.grib"
        assert obj.archive_requests["path"] == "out_w0.json"

    def test_deep_nested_suffix_applied(self):
        obj = self._make_nested_cls("archive_requests.sub.path")(
            archive_requests={"sub": {"path": "out.json"}}, **{"_parallel-output-suffix": "_w0"}
        )
        assert obj.archive_requests["sub"]["path"] == "out_w0.json"

    def test_nested_no_suffix_unchanged(self):
        obj = self._make_nested_cls("archive_requests.path")(archive_requests={"path": "out.json"})
        assert obj.archive_requests["path"] == "out.json"

    def test_nested_missing_leaf_key_ignored(self):
        obj = self._make_nested_cls("archive_requests.path")(
            archive_requests={"other": "value"}, **{"_parallel-output-suffix": "_w0"}
        )
        assert obj.archive_requests == {"other": "value"}

    def test_nested_missing_parent_key_ignored(self):
        obj = self._make_nested_cls("archive_requests.path")(**{"_parallel-output-suffix": "_w0"})
        assert obj.archive_requests is None

    def test_nested_parent_not_dict_ignored(self):
        obj = self._make_nested_cls("archive_requests.path")(
            archive_requests="not-a-dict", **{"_parallel-output-suffix": "_w0"}
        )
        assert obj.archive_requests == "not-a-dict"

    def test_nested_leaf_none_not_modified(self):
        obj = self._make_nested_cls("archive_requests.path")(
            archive_requests={"path": None}, **{"_parallel-output-suffix": "_w0"}
        )
        assert obj.archive_requests["path"] is None
