# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import math
import multiprocessing as mp
import os
import traceback
from collections.abc import Sequence
from typing import Any

from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.types import State

from ..output import ForwardOutput
from ..output import Output
from . import create_output
from . import output_registry

LOG = logging.getLogger(__name__)

_SIGNAL="TERMINATE"
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


def _sanitize_state(state: State) -> State:
    """Remove private keys and convert tensors so the state is safe to pickle."""
    clean = {k: v for k, v in state.items() if not k.startswith("_")}
    return _detach_tensors(clean)


def _split_state(state: State, num_chunks: int) -> list[State]:
    """Split a state into *num_chunks* chunks along the fields dimension.

    All non-field entries are replicated across chunks.
    """
    if "fields" not in state:
        raise ValueError("State dictionary must contain 'fields' key")
    if num_chunks <= 1:
        return [state]

    fields = state["fields"]
    fields_per_chunk = math.ceil(len(fields) / num_chunks)
    items = list(fields.items())

    chunks: list[State] = []
    for i in range(num_chunks):
        start = i * fields_per_chunk
        end = min(start + fields_per_chunk, len(items))
        if start >= len(items):
            break
        chunk = state.copy()
        chunk["fields"] = dict(items[start:end])
        chunks.append(chunk)

    return chunks


# ── ParallelOutput ───────────────────────────────────────────────────


@output_registry.register("parallel")
class ParallelOutput(Output):
    """Wraps another :class:`Output` and offloads ``write_step`` calls to
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
        output: Output | Any,
        num_writers: int | None = None,
        **kwargs: Any,
    ):
        """Initialise the ParallelOutput.

        Parameters
        ----------
        context : Context
            The inference context.
        metadata : Metadata
            Metadata for the dataset.
        output : Output | Any
            The inner output (or its config dict) that will be forked into
            writer processes.
        num_writers : int, optional
            Number of writer processes to spawn.  
            If not specified, an error is raised.
        **kwargs : Any
            Extra keyword arguments forwarded to :class:`ForwardOutput`.
        """
        super().__init__(context, metadata, **kwargs)

        assert num_writers is not None, "Error. ParallelOutput selected but no 'num_writers' specified in config."
        self.num_writers = num_writers

        # store the output config for printing and for creating outputs in the writer processes.
        self.output_config = output 

        # Writers are spawned in open() rather than here, because at
        # __init__ time the context may not yet have lead_time / time_step set.
        # Forking before those are populated would give the children stale copies.
        self._writers_running = False

    def open(self, state: State) -> None:
        """Spawn writer processes now that the context is fully initialised.

        This is called by the runner *after* lead_time and time_step have been
        set on the context, so the forked children will inherit a valid context.
        """
        if not self._writers_running:
            self._spawn_writers(self.context, self.output_config)
            self._writers_running = True
        # self.write_state(state) # Do I need to write the initial state here?
        pass


    # Cannot be an abstract method but should not be called directly on ParallelOutput.
    def write_step(self, state: State) -> None:
        return ValueError("ParallelOutput does not support write_step directly — it dispatches to writer processes. Make sure to call write_state instead.")

    def write_state(self, state: State) -> None:
        """Write the state, dispatching to writer processes when enabled."""
        step = state["step"]
        if self.output_frequency is not None:
            if (step % self.output_frequency).total_seconds() != 0:
                return

        # Parallel case — split state into chunks and send to writers via queues
        state_chunks = _split_state(state, self.num_writers)
        for i, chunk in enumerate(state_chunks):
            if not self._processes[i].is_alive():
                LOG.debug("Writer %d is dead", i)
                continue
            self._queues[i].put(_sanitize_state(chunk))

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

    # ── internals ───────────────────────────────────────────────────────

    def _spawn_writers(self, context: Context, output_config) -> None:
        """Fork writer processes.
        
        'self.num_writers' writer processes are spawned, each with its own queue for receiving states to write. 
        The writer processes run the '_writer_loop' method, which listens for incoming states on the queue and 
        calls 'write_step' on the wrapped output.
        """
        self._processes: list[mp.Process] = []
        self._queues: list[mp.Queue] = []

        for _ in range(self.num_writers):
            self._queues.append(mp.Queue())

        mp.set_start_method("fork", force=True)
        parent_pid = os.getpid()
        for i in range(self.num_writers):
            process = mp.Process(
                target=self._writer_loop,
                args=(i, self._queues[i], context,output_config),
                name=f"w{i}_for_p{parent_pid}",
            )
            process.start()
            self._processes.append(process)

        LOG.info("ParallelOutput: spawned %d writer processes", self.num_writers)

    def _writer_loop(self, writer_id: int, queue: mp.Queue, context: Context, output_config) -> None:
        """Event loop executed inside each forked writer process.
        
        Each writer process runs this loop, which listens for incoming messages on its queue.
        Messages can be either state dictionaries to write (which are passed to the wrapped output's 'write_step' method) or a "TERMINATE" signal to shut down the writer.

        Writers create their own output instance and write the initial state before entering the loop.

        """
        LOG.info("Writer %d started", writer_id)

        output = create_output(context, output_config, self.metadata, suffix=f"_w{writer_id}")
        has_written_initial_state = False

        while True:
            # Receive a message from the main process
            try:
                message = queue.get()
            except mp.queues.Empty:
                continue
            except Exception as e:
                LOG.error("Writer %d queue error: %s\n%s", writer_id, e, traceback.format_exc())
                break

            # Check for termination signal
            if message == _SIGNAL:
                LOG.debug("Writer %d received '%s'", writer_id, _SIGNAL)
                output.close()
                break

            # Otherwise, post process and write the received state using the wrapped output
            try:
                message = self.post_process(message)

                if not has_written_initial_state:
                    # Some outputs (e.g. netcdf) must be explictly opened.
                    if hasattr(output, "open"):
                        LOG.debug("Writer %d openning %s", writer_id, output)
                        output.open(message)
                    # each process needs to write initial state to do some setup
                    output.write_initial_state(message)
                    has_written_initial_state = True

                LOG.debug("Writer %d writing: %s", writer_id, message.get("date"))
                output.write_step(message)
                LOG.debug("Writer %d done: %s", writer_id, message.get("date"))
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
                    queue.put_nowait(_SIGNAL)
                else:
                    LOG.warning("Writer %d already dead", i)
            except Exception as e:
                LOG.error("Failed to send '%s' to writer %d: %s", _SIGNAL, i, e)

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
