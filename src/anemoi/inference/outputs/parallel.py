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
class ParallelOutput(ForwardOutput):
    """Wraps another :class:`Output` and offloads ``write_step`` calls to
    one or more forked writer processes.

    The wrapped output's ``per_writer_init`` hook (if any) is invoked once
    per worker so that the concrete output can adjust itself (e.g. open a
    different file per writer).

    Usage in YAML::

        output:
          parallel:
            num_writers: 2
            output:
              grib:
                path: output.grib
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
            Number of writer processes to spawn.  Falls back to
            ``context.writers_per_device`` if not given, and to ``0``
            (synchronous) if neither is set.
        **kwargs : Any
            Extra keyword arguments forwarded to :class:`ForwardOutput`.
        """
        super().__init__(context, metadata, output=output, **kwargs)

        assert num_writers is not None, "Error. ParallelOutput selected but no 'num_writers' specified in config."
        self.num_writers = num_writers

        self._writers_running = False


    def write_state(self, state: State) -> None:
        """Write the state, dispatching to writer processes when enabled."""
        step = state["step"]
        if self.output_frequency is not None:
            if (step % self.output_frequency).total_seconds() != 0:
                return

        state = self.post_process(state)

        if self.num_writers <= 0:
            # Sequntial fallback — delegate to wrapped output
            self.output.write_step(self.modify_state(state))
            return

        # Parallel path
        if not self._writers_running:
            self._spawn_writers()
            self._writers_running = True

        state = self.modify_state(state)
        state_chunks = _split_state(state, self.num_writers)
        for i, chunk in enumerate(state_chunks):
            self._queues[i].put(_sanitize_state(chunk))

    def close(self) -> None:
        """Terminate writer processes, then close the wrapped output."""
        if self._writers_running:
            self._terminate_all_writers()
        self.output.close()

    def __repr__(self) -> str:
        return f"ParallelOutput(num_writers={self.num_writers}, output={self.output})"

    def print_summary(self, depth: int = 0) -> None:
        LOG.info(
            "%sParallelOutput: num_writers=%d",
            " " * depth,
            self.num_writers,
        )
        self.output.print_summary(depth + 1)

    # ── internals ───────────────────────────────────────────────────────

    def _spawn_writers(self) -> None:
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
                args=(i, self._queues[i]),
                name=f"w{i}_for_p{parent_pid}",
            )
            process.start()
            self._processes.append(process)

        LOG.info("ParallelOutput: spawned %d writer processes", self.num_writers)

    def _writer_loop(self, writer_id: int, queue: mp.Queue) -> None:
        """Event loop executed inside each forked writer process.
        
        Each writer process runs this loop, which listens for incoming messages on its queue.
        Messages can be either state dictionaries to write (which are passed to the wrapped output's 'write_step' method) or a "TERMINATE" signal to shut down the writer.

        
        If the wrapped output has a 'per_writer_init' method, it is called once at the start of the loop to allow for any necessary initialization 
        (e.g., appending a writer ID to the output file name)."""
        LOG.info("Writer %d started", writer_id)

        # Allow the wrapped output to perform any per-writer initialization (e.g. open a different file per writer)
        if hasattr(self.output, "per_writer_init"):
            self.output.per_writer_init(writer_id)

        while True:
            try:
                message = queue.get(timeout=1)
            except mp.queues.Empty:
                continue
            except Exception as e:
                LOG.error("Writer %d queue error: %s\n%s", writer_id, e, traceback.format_exc())
                break

            if message == "TERMINATE":
                LOG.debug("Writer %d received TERMINATE", writer_id)
                break

            try:
                LOG.debug("Writer %d writing: %s", writer_id, message.get("date"))
                self.output.write_step(message)
                LOG.debug("Writer %d done: %s", writer_id, message.get("date"))
            except Exception as e:
                LOG.error("Writer %d write error: %s\n%s", writer_id, e, traceback.format_exc())
                break

        LOG.info("Writer %d shutting down", writer_id)

    def _terminate_all_writers(self) -> None:
        """Gracefully shut down all writer processes.
        Each writer process is sent a "TERMINATE" message via its queue, then joined with a timeout. If any writer fails to exit within the timeout, it is forcefully terminated.
        """
        self._writers_running = False
        LOG.info("ParallelOutput: shutting down writers...")

        for i, queue in enumerate(self._queues):
            try:
                queue.put_nowait("TERMINATE")
            except Exception as e:
                LOG.error("Failed to send TERMINATE to writer %d: %s", i, e)

        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                LOG.warning("Writer %s did not exit — terminating", p.name)
                p.terminate()

        LOG.info("ParallelOutput: all writers terminated.")
