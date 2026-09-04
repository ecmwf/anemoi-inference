# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
import logging
import math
import multiprocessing as mp
import os
import traceback
from collections.abc import Callable
from collections.abc import Generator
from enum import Enum
from functools import cache
from time import sleep
from typing import Any
from typing import Literal
from typing import get_args

from anemoi.transform.variables import Variable

from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.types import State

from ..output import Output
from . import create_output
from . import output_registry

LOG = logging.getLogger(__name__)


# ── helpers ─────────────────────────────────────────────────────────────
def _detach_tensors(obj: Any) -> Any:
    """Recursively convert torch tensors to numpy arrays.

    This prevents "Cannot re-initialize CUDA in forked subprocess" errors
    when pickling state dicts for the multiprocessing queue.
    """
    try:
        import torch
    except ImportError:
        return obj

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: _detach_tensors(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_detach_tensors(v) for v in obj)
    return obj


def _template_message_bytes(field: Any) -> bytes:
    """Return the raw GRIB bytes of *field* stripped of its data section.

    The writer only needs the template's metadata (grid, packing, product
    definition); the values are discarded and overwritten. With this we can
    reduce what crosses the fork boundary from hundreds of KB to ~200 B per
    template by overwriting the values with zeros.

    ``bitsPerValue`` is forced back to its original value after zeroing the
    data section: eccodes otherwise resets it to the default (typically 24)
    when the section is rewritten with a constant field, which would change
    the packing of every message the writer produces vs. the non-parallel
    path.

    Falls back to :meth:`field.message` if the shrink cannot be applied.
    """
    import numpy as np

    try:
        handle = field.handle.clone()
        bpv = handle.get("bitsPerValue")
        n = int(handle.get("numberOfDataPoints"))
        handle.set_values(np.zeros(n))
        handle.set("bitsPerValue", bpv)
        return handle.get_buffer()
    except Exception as e:
        LOG.debug("Could not strip data section from GRIB template: %s; sending full message.", e)
        return field.message()


def _serialise_grib_templates(templates: dict) -> dict[str, bytes]:
    """Convert a dict of earthkit GRIB fields to a dict of raw GRIB bytes so they can be pickled.

    Each template is stripped of its data section before pickling (see
    :func:`_template_message_bytes`) so the number of bytes shipped through the
    multiprocessing queue is a few hundred per template instead of the full
    field size.

    Templates that cannot be converted (e.g. non-GRIB fields, or future earthkit
    wrappers that do not expose ``.message()``) are skipped with a warning so a
    single unserialisable template does not abort the whole state; the writer
    will simply not receive that template and fall back to whatever the
    template manager provides.
    """
    result = {}
    for name, field in templates.items():
        try:
            result[name] = _template_message_bytes(field)
        except Exception as e:
            LOG.warning(
                "Could not serialise GRIB template for '%s' (type=%s): %s; skipping.",
                name,
                type(field).__name__,
                e,
                exc_info=True,
            )
    return result


def _deserialise_grib_templates(bytes_templates: dict[str, bytes]) -> dict[str, Any]:
    """Reconstruct earthkit GRIB fields from raw bytes.

    Uses ``earthkit.data.from_source("memory", ...)`` so the writer processes see
    the exact same field type they would see in the non-parallel path (i.e. the
    field the input pipeline stored under ``_grib_templates_for_output``).
    Failed reconstructions are skipped with a warning so a single bad template
    does not abort the whole state.
    """
    import earthkit.data as ekd

    result: dict[str, Any] = {}
    for name, msg in bytes_templates.items():
        try:
            result[name] = ekd.from_source("memory", msg)[0]
        except Exception as e:
            LOG.warning("Could not reconstruct GRIB template for '%s' from bytes: %s", name, e)
    return result


def _sanitise_state(state: State, grib_templates_bytes: dict[str, bytes] | None = None) -> State:
    """Remove private keys and convert tensors so the state is safe to pickle.

    ``grib_templates_bytes``, if provided, replaces ``_grib_templates_for_output``
    on the returned state with the corresponding ``_grib_templates_bytes_for_output``
    payload. Callers precompute this once per dispatch so we do not re-serialise the
    same templates dict N times when there are N writers.
    """

    unpicklable_keys = ["_input", "_variables"]

    state = state.copy()

    if grib_templates_bytes is not None:
        state["_grib_templates_bytes_for_output"] = grib_templates_bytes
    state.pop("_grib_templates_for_output", None)

    for key in unpicklable_keys:
        if state.get(key) is not None:
            state.pop(key)
            LOG.debug("Removed unpicklable key '%s' from state before sending to writer process", key)
    return _detach_tensors(state)


def _restore_grib_templates(state: State) -> State:
    """Reverse of the GRIB-template step in :func:`_sanitise_state`.

    Called inside writer processes so the wrapped output sees the same
    ``_grib_templates_for_output`` layout it would in single-process mode.
    No-op if the state does not carry serialised templates.
    """
    if not isinstance(state, dict):
        return state
    bytes_templates = state.pop("_grib_templates_bytes_for_output", None)
    if not bytes_templates:
        return state
    state["_grib_templates_for_output"] = _deserialise_grib_templates(bytes_templates)
    return state


CHUNK_STRATEGIES = Literal["by_worker", "by_metadata", "by_size"]
VALID_CHUNK_STRATEGIES = get_args(CHUNK_STRATEGIES)


class Chunker:
    """Chunking strategies for dividing the state into smaller parts for parallel outputting."""

    def __init__(self, typed_variables: dict[str, Variable], num_writers: int):
        """Chunker

        Parameters
        ----------
        typed_variables : dict[str, Variable]
            Dictionary of field name to its corresponding Variable metadata.
        """
        self.typed_variables = typed_variables
        self.num_writers = num_writers

    @cache
    def _grouped_fields_by_metadata(self, keys: tuple[str], max_groups: int = -1) -> list[str]:
        grouped_fields = {}
        for field_name, meta in self.typed_variables.items():
            key_tuple = tuple(getattr(meta, key, None) for key in keys)
            grouped_fields.setdefault(key_tuple, []).append(field_name)
        if len(grouped_fields) == 1:
            LOG.warning(
                "All fields have the same metadata for keys %s. Consider using different keys for chunking.", keys
            )
        else:
            LOG.info(
                "Fields have been grouped into %d distinct metadata combinations for keys %s.",
                len(grouped_fields),
                keys,
            )

        grouped_fields_list = list(grouped_fields.values())

        if max_groups > 0 and len(grouped_fields) > max_groups:
            new_grouped_fields = {}
            for i, fields in enumerate(grouped_fields_list):
                new_key = i % max_groups
                new_grouped_fields.setdefault(new_key, []).extend(fields)
            grouped_fields_list = list(new_grouped_fields.values())

        return grouped_fields_list

    def by_metadata(self, keys: list[str], max_groups: int = -1) -> Callable[[State], Generator[State, None, None]]:
        """Chunk the state into smaller parts based on the specified metadata keys,
        ensuring that variables with the same key=value pair are kept together in the same chunk.
        """

        def chunker(state: State) -> Generator[State, None, None]:
            fields = state["fields"]
            grouped_fields = self._grouped_fields_by_metadata(tuple(keys), max_groups=max_groups)

            for group_keys in grouped_fields:
                chunk = state.copy()
                chunk["fields"] = {k: fields[k] for k in group_keys if k in fields}
                yield chunk

        return chunker

    def by_size(self, fields_per_chunk: int) -> Callable[[State], Generator[State, None, None]]:
        """Chunk the state into smaller parts, each containing a specified number of fields."""

        def chunker(state: State) -> Generator[State, None, None]:
            fields = state["fields"]
            num_fields = len(fields)
            for start in range(0, num_fields, fields_per_chunk):
                stop = start + fields_per_chunk
                chunk = state.copy()
                fields_subset = itertools.islice(fields.items(), start, stop)
                chunk["fields"] = dict(fields_subset)
                yield chunk

        return chunker

    def by_worker(self) -> Callable[[State], Generator[State, None, None]]:
        """Chunk the state into smaller parts, one for each worker."""

        def chunker(state: State) -> Generator[State, None, None]:
            fields = state["fields"]
            fields_per_chunk = math.ceil(len(fields) / self.num_writers)

            for i in range(self.num_writers):
                chunk = state.copy()

                start = i * fields_per_chunk
                stop = start + fields_per_chunk

                # copy the subset of the fields into a new state dict
                fields_subset = itertools.islice(fields.items(), start, stop)
                chunk["fields"] = dict(fields_subset)
                yield chunk

        return chunker


class MessageType(str, Enum):
    """Types of messages sent from the main process to the writer processes. Used for logging and control flow in the writer loop."""

    TERMINATE = "Terminate"
    INITIAL_STATE = "InitialState"
    STATE = "State"
    OPEN = "Open"


# ── ParallelOutput ───────────────────────────────────────────────────


@output_registry.register("parallel")
class ParallelOutput(Output):
    """Wraps another :class:`Output` and offloads ``write_state`` calls to
    one or more forked writer processes. The output is split along the field dimension
    and each chunk is sent to a different writer process via multiprocessing queues.
    Each writer process writes the initial state into its own file.

    When writing a file output, a suffix '_{writer_id}' is appended to the file name to avoid conflicts between writers.

    Usage in YAML::

        output:
          parallel:
            num_writers: 2
            output:
              grib:
                path: output.grib

    This yaml will result in the following outputs being written:
        - output_w0.grib
        - output_w1.grib

    """

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        output: Output | Any | None = None,
        num_writers: int = 1,
        chunk_strategy: CHUNK_STRATEGIES | dict[CHUNK_STRATEGIES, dict[str, Any]] = "by_worker",
        **kwargs: Any,
    ):
        """Initialise the ParallelOutput.

        Parameters
        ----------
        context : Context
            The inference context.
        metadata : Metadata
            Metadata for the dataset.
        output : Output | Any | None
            The inner output (or its config dict) that will be forked into
            writer processes.
        num_writers : int
            Number of writer processes to spawn.
            Must be >= 1.
            Defaults to 1 (single output file, asynchronous writes).
        chunk_strategy : CHUNK_STRATEGIES | dict[CHUNK_STRATEGIES, dict[str, Any]], default="by_worker"
            The strategy for chunking the output among writer processes.
            Can be a string (one of "by_size", "by_metadata", "by_worker") or a dictionary
            with a single key being the strategy name and the value being a dictionary of
            keyword arguments for that strategy.
            - `by_worker`, no additional arguments are needed.
            - `by_size`, requires an additional argument `fields_per_chunk` specifying the number of fields per chunk.
            - `by_metadata`, requires an additional argument `keys` specifying a list of metadata keys to keep in a chunk.
        **kwargs : Any
            Forwarded to the inner output.
        """
        super().__init__(
            context,
            metadata,
        )

        self.num_writers = int(num_writers)
        if self.num_writers < 1:
            raise ValueError("num_writers must be at least 1")

        # store the output config for printing and for creating outputs in the writer processes.
        self.kwargs = {}
        if output is None:
            output = kwargs
        else:
            # pass the kwargs to the writer processes
            self.kwargs = kwargs

        self.output_config = output

        # Writers are spawned in open() rather than here, because at
        # __init__ time the context may not yet have lead_time / time_step set.
        self._writers_running = False

        # Cache of the serialised GRIB templates bytes-map, keyed by id() of the
        # ``_grib_templates_for_output`` dict on the incoming state. Populated in
        # dispatch_state_to_writers so we don't re-serialise once per writer
        # (or once per rollout step, when the input pipeline reuses the same dict).
        self._grib_templates_bytes_cache_key: int | None = None
        self._grib_templates_bytes_cache_value: dict[str, bytes] | None = None

        # Chunking strategy for dividing work among writer processes.
        if not isinstance(chunk_strategy, (str, dict)):
            raise ValueError("chunk_strategy must be a string or a dictionary")
        if isinstance(chunk_strategy, dict) and len(chunk_strategy) != 1:
            raise ValueError("chunk_strategy dictionary must have exactly one key")

        chunk_strategy_name = chunk_strategy if isinstance(chunk_strategy, str) else next(iter(chunk_strategy.keys()))
        chunk_strategy_init = next(iter(chunk_strategy.values())) if isinstance(chunk_strategy, dict) else {}
        chunker = Chunker(self.metadata.typed_variables, self.num_writers)

        match chunk_strategy_name:
            case "by_size":
                assert isinstance(chunk_strategy, dict)
                chunking_func = chunker.by_size(**chunk_strategy_init)
            case "by_metadata":
                assert isinstance(chunk_strategy, dict)
                chunking_func = chunker.by_metadata(**chunk_strategy_init)
            case "by_worker":
                chunking_func = chunker.by_worker()
            case _:
                raise ValueError(
                    f"Invalid chunk_strategy: {chunk_strategy_name}. Must be one of {VALID_CHUNK_STRATEGIES}"
                )

        self.chunking_func = chunking_func

    def open(self, state: State) -> None:
        """Spawn the writer processes during open() instead of __init__() to ensure they have access to the full context.

        Pass the open() message to the writers so that they can call the output-appropriate open() method.
        """
        if not self._writers_running:
            self._spawn_writers(self.context, self.output_config, **self.kwargs)
            self._writers_running = True
        self.dispatch_state_to_writers(state, message=MessageType.OPEN)

    # Cannot be an abstract method but should not be called directly on ParallelOutput.
    def write_step(self, state: State) -> None:
        raise ValueError(
            "ParallelOutput does not support write_step directly — it dispatches to writer processes. Make sure to call write_state instead."
        )

    def _check_writer_alive(self, writer_id) -> None:
        """Raise an error if a writer process has died."""
        process = self._processes[writer_id]
        if not process.is_alive():
            # prevents hanging on cleanup if a writer had an error during runtime by draining the queues of any unconsumed messages.
            for queue in self._queues:
                queue.cancel_join_thread()
            raise RuntimeError(
                f"Writer {writer_id} is dead, inference will now fail. Check previous logs for errors in the writer process."
            )

    def write_state(self, state: State, message=MessageType.STATE) -> None:
        """Write the state, dispatching to writer processes when enabled."""
        self.dispatch_state_to_writers(state, message=message)

    def dispatch_state_to_writers(self, state: State, message=MessageType.STATE) -> None:
        """Send the state to each writer process via multiprocessing queues.

        Takes an optional 'message' argument to indicate the type of message being sent, which is used for control flow in the writer loop.
        """
        grib_templates_bytes = self._get_or_serialise_grib_templates(state)

        for i, chunk in enumerate(self.chunking_func(state)):
            worker_id = i % self.num_writers
            self._check_writer_alive(worker_id)
            self._queues[worker_id].put((_sanitise_state(chunk, grib_templates_bytes), message))

    def _get_or_serialise_grib_templates(self, state: State) -> dict[str, bytes] | None:
        """Return the serialised GRIB templates bytes-map, serialising it on the first call.

        Result is memoised on the instance and keyed by ``id()`` of the
        ``_grib_templates_for_output`` dict: as long as the input pipeline hands
        us the same dict object across dispatches, we serialise once and reuse
        the bytes for every writer and every subsequent step. When the pipeline
        replaces the dict (e.g. the set of templates changes), the id() mismatch
        forces a fresh serialisation.

        Returns None if the state has no GRIB templates to send.
        """
        templates = state.get("_grib_templates_for_output")
        if templates is None:
            return None
        key = id(templates)
        if key == self._grib_templates_bytes_cache_key:
            return self._grib_templates_bytes_cache_value
        bytes_map = _serialise_grib_templates(templates)
        LOG.debug("Serialised %d GRIB templates to bytes for writer processes", len(bytes_map))
        self._grib_templates_bytes_cache_key = key
        self._grib_templates_bytes_cache_value = bytes_map
        return bytes_map

    def close(self) -> None:
        """Terminate writer processes, then close the wrapped output."""
        if self._writers_running:
            self._terminate_all_writers()

    def __repr__(self) -> str:
        return f"ParallelOutput(num_writers={self.num_writers}, output={self.output_config})"

    def print_summary(self, depth: int = 0) -> None:
        LOG.info(
            "%sParallelOutput: num_writers=%d, output=%s",
            " " * depth,
            self.num_writers,
            self.output_config,
        )

    def write_initial_state(self, state: State) -> None:
        """Write the initial state."""
        self.dispatch_state_to_writers(state, message=MessageType.INITIAL_STATE)

    # ── internals ───────────────────────────────────────────────────────

    def _spawn_writers(self, context: Context, output_config, **kwargs) -> None:
        """Fork writer processes.

        'self.num_writers' writer processes are spawned, each with its own queue for receiving states to write.
        The writer processes run the '_writer_loop' method, which listens for incoming states on the queue and
        calls 'write_state' on the wrapped output.
        """
        self._processes: list[mp.Process] = []
        self._queues: list[mp.Queue] = []

        ctx = mp.get_context("fork")
        for _ in range(self.num_writers):
            # Use a bounded queue to prevent unlimited memory growth if the writers can't keep up with the main process
            self._queues.append(ctx.Queue(maxsize=10))

        parent_pid = os.getpid()
        for i in range(self.num_writers):
            process = ctx.Process(
                target=self._writer_loop,
                args=(i, self._queues[i], context, output_config),
                kwargs=kwargs,
                name=f"w{i}_for_p{parent_pid}",
            )
            process.start()
            self._processes.append(process)

        LOG.info("ParallelOutput: spawned %d writer processes", self.num_writers)

    def _writer_loop(self, writer_id: int, queue: mp.Queue, context: Context, output_config, **kwargs) -> None:
        """Event loop executed inside each forked writer process.

        Each writer process runs this loop, which listens for incoming messages on its queue.
        Messages are a tuple of (content, message_type), where 'message_type' indicates the type of message (e.g., MessageType.STATE, MessageType.INITIAL_STATE, MessageType.TERMINATE)
        and 'content' is the state dictionary to write (for STATE and INITIAL_STATE messages).

        Writers create their own output instance inside the writer process.

        """
        LOG.info("Writer %d started", writer_id)

        output = create_output(
            context, output_config, self.metadata, **kwargs, **{"_parallel-output-suffix": f"_w{writer_id}"}
        )

        while True:
            # Receive a message from the main process
            try:
                message, message_type = queue.get()
            except Exception as e:
                LOG.error("Writer %d queue error: %s\n%s", writer_id, e, traceback.format_exc())
                break

            try:
                # read message type to determine how to process the message
                LOG.debug("Writer %d received '%s'", writer_id, message_type)
                if message_type == MessageType.TERMINATE:
                    output.close()
                    break
                elif message_type == MessageType.OPEN:
                    output.open(_restore_grib_templates(message))
                elif message_type == MessageType.INITIAL_STATE:
                    output.write_initial_state(_restore_grib_templates(message))
                elif message_type == MessageType.STATE:
                    LOG.debug("Writer %d writing: %s", writer_id, message.get("date"))
                    output.write_state(_restore_grib_templates(message))
                    LOG.debug("Writer %d done: %s", writer_id, message.get("date"))
                else:
                    LOG.warning("Writer %d received message with unknown type '%s'", writer_id, message_type)
            except Exception as e:
                LOG.error("Writer %d write error: %s\n%s", writer_id, e, traceback.format_exc())
                break

        LOG.info("Writer %d shutting down", writer_id)

    def _terminate_all_writers(self, timeout_s=10) -> None:
        """Gracefully shut down all writer processes.

        Sends a TERMINATE sentinel to each writer queue. Writers consume all
        pending messages before reaching the sentinel, so no queued data is lost.
        After sending, we join each process and only force-terminate if it hangs
        beyond the timeout (which should not happen under normal conditions).
        """
        self._writers_running = False
        LOG.info("ParallelOutput: shutting down writers...")

        # Send the shutdown signal to each writer.
        # It will only be consumed after all subsequent messages, to ensure no data loss.
        for i, queue in enumerate(self._queues):
            try:
                if self._processes[i].is_alive():
                    queue.put(("", MessageType.TERMINATE))
                else:
                    LOG.warning("Writer %d already dead", i)
            except Exception as e:
                LOG.error("Failed to send '%s' to writer %d: %s", MessageType.TERMINATE, i, e)

        LOG.info("ParallelOutput: waiting for writers to finish and exit...")

        sleep(timeout_s)

        forcibly_teminated = False
        for i, process in enumerate(self._processes):
            if process.is_alive():
                LOG.warning(
                    "Writer %d did not exit within timeout, forcefully terminating - Data yet be written will be lost",
                    i,
                )
                process.terminate()
                process.join()
                forcibly_teminated = True

        LOG.info("ParallelOutput: all writers terminated.")
        if forcibly_teminated:
            raise RuntimeError(
                "One or more writers were forcefully terminated after exceeding the shutdown timeout of %s. This can result in forecast data loss. Please check the logs for more details. Consider increasing the number of writer processes to avoid this.",
                timeout_s,
            )
