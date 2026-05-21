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
from enum import Enum
from typing import Any

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


def _sanitise_state(state: State) -> State:
    """Remove private keys and convert tensors so the state is safe to pickle."""

    unpicklable_keys = ["_grib_templates_for_output", "_input"]

    for key in unpicklable_keys:
        if state.get(key) is not None:
            state.pop(key)
    return _detach_tensors(state)


def _get_state_chunk(state: State, num_chunks: int, index: int) -> State:
    """Get a specific chunk of a state along the fields dimension."""
    if "fields" not in state:
        raise ValueError("State dictionary must contain 'fields' key")
    assert 0 <= index < num_chunks, f"Index {index} out of range for {num_chunks} chunks"
    if num_chunks <= 1:
        return state

    # determine the subset of fields for this chunk
    fields = state["fields"]
    fields_per_chunk = math.ceil(len(fields) / num_chunks)
    start = index * fields_per_chunk
    stop = start + fields_per_chunk

    # copy the subset of the fields into a new state dict
    fields_subset = itertools.islice(fields.items(), start, stop)
    chunk = state.copy()
    chunk["fields"] = dict(fields_subset)
    return chunk


class MessageType(str, Enum):
    """Types of messages sent from the main process to the writer processes. Used for logging and control flow in the writer loop."""

    TERMINATE = "Terminate"
    INITIAL_STATE = "InitialState"
    STATE = "State"


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
        **kwargs : Any
            Extra keyword arguments forwarded to :class:`Output`.
        """
        super().__init__(context, metadata)

        self.num_writers = num_writers
        assert self.num_writers >= 1, "num_writers must be at least 1"

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

    def open(self, state: State) -> None:
        """Spawn the writer processes during open() instead of __init__() to ensure they have access to the full context."""
        if not self._writers_running:
            self._spawn_writers(self.context, self.output_config, **self.kwargs)
            self._writers_running = True
        pass

    # Cannot be an abstract method but should not be called directly on ParallelOutput.
    def write_step(self, state: State) -> None:
        return ValueError(
            "ParallelOutput does not support write_step directly — it dispatches to writer processes. Make sure to call write_state instead."
        )

    def write_state(self, state: State, message=MessageType.STATE) -> None:
        """Write the state, dispatching to writer processes when enabled."""
        # Parallel case — split state into chunks and send to writers via queues
        for i in range(self.num_writers):
            if not self._processes[i].is_alive():
                raise RuntimeError(
                    f"Writer {i} is dead, inference will now fail. Check previous logs for errors in the writer process."
                )
            chunk = _get_state_chunk(state, self.num_writers, i)
            self._queues[i].put((_sanitise_state(chunk), message))

    def close(self) -> None:
        """Terminate writer processes, then close the wrapped output."""
        if self._writers_running:
            self._terminate_all_writers()

    def __repr__(self) -> str:
        return f"ParallelOutput(num_writers={self.num_writers}, output={self.output_config})"

    def print_summary(self, depth: int = 0) -> None:
        LOG.info(
            "%sParallelOutput: num_writers=%d",
            " " * depth,
            self.num_writers,
        )
        self.output.print_summary(depth + 1)

    def write_initial_state(self, state: State) -> None:
        """Write the initial state."""
        self.write_state(state, message=MessageType.INITIAL_STATE)

    # ── internals ───────────────────────────────────────────────────────

    def _spawn_writers(self, context: Context, output_config, **kwargs) -> None:
        """Fork writer processes.

        'self.num_writers' writer processes are spawned, each with its own queue for receiving states to write.
        The writer processes run the '_writer_loop' method, which listens for incoming states on the queue and
        calls 'write_state' on the wrapped output.
        """
        self._processes: list[mp.Process] = []
        self._queues: list[mp.Queue] = []

        for _ in range(self.num_writers):
            # Use a bounded queue to prevent unlimited memory growth if the writers can't keep up with the main process
            self._queues.append(mp.Queue(maxsize=10))

        mp.set_start_method("fork", force=True)
        parent_pid = os.getpid()
        for i in range(self.num_writers):
            process = mp.Process(
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

        output = create_output(context, output_config, self.metadata, suffix=f"_w{writer_id}", **kwargs)

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
                elif message_type == MessageType.INITIAL_STATE:
                    output.write_initial_state(message)
                elif message_type == MessageType.STATE:
                    LOG.debug("Writer %d writing: %s", writer_id, message.get("date"))
                    output.write_state(message)
                    LOG.debug("Writer %d done: %s", writer_id, message.get("date"))
                else:
                    LOG.warning("Writer %d received message with unknown type '%s'", writer_id, message_type)
            except Exception as e:
                LOG.error("Writer %d write error: %s\n%s", writer_id, e, traceback.format_exc())
                break

        LOG.info("Writer %d shutting down", writer_id)
        exit(0)

    def _terminate_all_writers(self) -> None:
        """Gracefully shut down all writer processes.
        Each writer process is sent a "TERMINATE" signal via its queue, then joined with a timeout. If any writer fails to exit within the timeout, it is forcefully terminated.
        """
        self._writers_running = False
        LOG.info("ParallelOutput: shutting down writers...")

        for i, queue in enumerate(self._queues):
            try:
                if self._processes[i].is_alive():
                    queue.put_nowait(("", MessageType.TERMINATE))
                else:
                    LOG.warning("Writer %d already dead", i)
            except Exception as e:
                LOG.error("Failed to send '%s' to writer %d: %s", MessageType.TERMINATE, i, e)

        LOG.info("ParallelOutput: waiting for writers to exit...")
        # Prevent hanging on cleanup if a writer had an error during runtime
        # by draining the queues of any unconsumed messages.
        for i, queue in enumerate(self._queues):
            queue.cancel_join_thread()
            try:
                while not queue.empty():
                    queue.get_nowait()
            except Exception:
                pass
            queue.close()

        LOG.info("ParallelOutput: all writers terminated.")
