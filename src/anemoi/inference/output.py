# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import datetime
import logging
import math
import multiprocessing as mp
import os
import traceback
from abc import ABC
from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any

from anemoi.inference.post_processors import create_post_processor
from anemoi.inference.processor import Processor
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State

if TYPE_CHECKING:
    from anemoi.inference.context import Context
    from anemoi.inference.metadata import Metadata

LOG = logging.getLogger(__name__)


class Output(ABC):
    """Abstract base class for output mechanisms."""

    def __init__(
        self,
        context: "Context",
        metadata: "Metadata",
        *,
        variables: list[str] | None = None,
        post_processors: list[ProcessorConfig] | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
        num_writers: int | None = None,
    ):
        """Initialize the Output object.

        Parameters
        ----------
        context : Context
            The context in which the output operates.
        metadata : Metadata
            Metadata corresponding to the dataset this output is handling.
        post_processors : Optional[List[ProcessorConfig]], default None
            Post-processors to apply to the output
        output_frequency : Optional[int], optional
            The frequency at which to output states, by default None.
        write_initial_state : Optional[bool], optional
            Whether to write the initial state, by default None.
        num_writers : Optional[int], optional
            Number of parallel writer processes to use per device.
            If 0 or None, writing is done synchronously in the main process.
        """
        self.context = context
        self.metadata = metadata
        self.dataset_name = metadata.dataset_name
        self.reference_date = context.reference_date

        self._post_processor_confs = post_processors or []

        self._write_step_zero = write_initial_state
        self._output_frequency = output_frequency

        self.variables = variables
        if self.variables is not None:
            if not isinstance(self.variables, (list, tuple)):
                self.variables = [self.variables]

        self.typed_variables = self.metadata.typed_variables.copy()
        self.typed_variables.update(self.context.typed_variables)

        # Parallel writer configuration
        if num_writers is not None:
            self.num_writers = num_writers
        else:
            self.num_writers = getattr(self.context, "writers_per_device", 0) or 0
        self._writers_running = False

    def skip_variable(self, variable: str) -> bool:
        """Check if a variable should be skipped.

        Parameters
        ----------
        variable : str
            The variable to check.

        Returns
        -------
        bool
            True if the variable should be skipped, False otherwise.
        """
        return self.variables is not None and variable not in self.variables

    @cached_property
    def post_processors(self) -> list[Processor]:
        """Return post-processors."""

        processors = []

        for processor in self._post_processor_confs:
            processors.append(create_post_processor(self.context, processor, self.metadata))

        return processors

    def post_process(self, state: State) -> State:
        """Apply post processors to the state.

        Parameters
        ----------
        state : State
            The state.

        Returns
        -------
        State
            The processed state.
        """
        for processor in self.post_processors:
            LOG.info("Post processor: %s", processor)
            state = processor.process(state)
        return state

    def __repr__(self) -> str:
        """Return a string representation of the Output object.

        Returns
        -------
        str
            String representation of the Output object.
        """
        return f"{self.__class__.__name__}()"

    def write_initial_state(self, state: State) -> None:
        """Write the initial state.

        Parameters
        ----------
        state : State
            The initial state to write.
        """
        state.setdefault("step", datetime.timedelta(0))
        if self.write_step_zero:
            self.write_step(self.post_process(state))

    def write_state(self, state: State) -> None:
        """Write the state.

        Parameters
        ----------
        state : State
            The state to write.
        """
        step = state["step"]
        if self.output_frequency is not None:
            if (step % self.output_frequency).total_seconds() != 0:
                return

        state = self.post_process(state)

        # seq writing path
        if self.num_writers == 0:
            return self.write_step(state)

        # Parallel writing path

        # spawn writers if not already running
        # can't spawn during initialisation
        if not self._writers_running:
            self._spawn_writers()
            self._writers_running = True


        state_chunks = self._split_state(state, self.num_writers)
        for i, chunk in enumerate(state_chunks):
            # Remove all private keys (prefixed with '_') as they may contain
            # unpicklable objects (e.g. _input, _grib_templates_for_output, _mask, etc.)
            chunk = {k: v for k, v in chunk.items() if not k.startswith("_")}
            # Convert any remaining CUDA/torch tensors to numpy to avoid
            # "Cannot re-initialize CUDA in forked subprocess" errors
            chunk = self._detach_tensors(chunk)
            self._queues[i].put(chunk)

    @classmethod
    def reduce(cls, state: State) -> State:
        """Create a new state which is a projection of the original state on the last step in the multi-steps dimension.

        Parameters
        ----------
        state : State
            The original state.

        Returns
        -------
        State
            The reduced state.
        """
        reduced_state = state.copy()
        reduced_state["fields"] = {}
        for field, values in state["fields"].items():
            if len(values.shape) > 1:
                reduced_state["fields"][field] = values[-1, :]
            else:
                reduced_state["fields"][field] = values
        return reduced_state

    def open(self, state: State) -> None:
        """Open the output for writing.

        Parameters
        ----------
        state : State
            The state to open.
        """
        # Override this method when initialisation is needed
        pass

    def close(self) -> None:
        """Close the output."""
        return
        if self.num_writers > 0 and self._writers_running:
            self._terminate_all_writers()

    @abstractmethod
    def write_step(self, state: State) -> None:
        """Write a step of the state.

        Parameters
        ----------
        state : State
            The state to write.
        """
        pass

    @cached_property
    def write_step_zero(self) -> bool:
        """Determine whether to write the initial state."""
        if self._write_step_zero is not None:
            return self._write_step_zero

        return self.context.write_initial_state

    @cached_property
    def output_frequency(self) -> datetime.timedelta | None:
        """Get the output frequency."""
        from anemoi.utils.dates import as_timedelta

        if self._output_frequency is not None:
            return as_timedelta(self._output_frequency)

        if self.context.output_frequency is not None:
            return as_timedelta(self.context.output_frequency)

        return None

    def print_summary(self, depth: int = 0) -> None:
        """Print a summary of the output configuration.

        Parameters
        ----------
        depth : int, optional
            The indentation depth for the summary, by default 0.
        """
        LOG.info(
            "%s%s: output_frequency=%s write_initial_state=%s num_writers=%s",
            " " * depth,
            self,
            self.output_frequency,
            self.write_step_zero,
            self.num_writers,
        )

    # ── Parallel writer infrastructure ──────────────────────────────────

    def _spawn_writers(self) -> None:
        """Spawn parallel writer processes."""
        self._processes: list[mp.Process] = []
        self._queues: list[mp.Queue] = []
        self._stop_event = mp.Event()

        for _ in range(self.num_writers):
            self._queues.append(mp.Queue())

        mp.set_start_method("fork", force=True)
        parent_pid = os.getpid()
        for i in range(self.num_writers):
            process = mp.Process(
                target=self._writer_function,
                args=(i, self._queues[i], self._stop_event),
                name=f"w{i}_for_p{parent_pid}",
            )
            process.start()
            self._processes.append(process)

        LOG.info("Spawned %d writer processes", self.num_writers)

    def _writer_function(self, writer_id: int, queue: mp.Queue, stop_event: mp.Event) -> None:
        """Worker function running in each writer process.

        Parameters
        ----------
        writer_id : int
            Unique identifier for this writer.
        queue : mp.Queue
            Queue to receive state chunks from the main process.
        stop_event : mp.Event
            Event signalling that writing should stop.
        """
        LOG.info("Writer %d started and waiting for messages...", writer_id)
        self.per_writer_init(writer_id)

        while not stop_event.is_set():
            try:
                message = queue.get(timeout=1)

                LOG.info("Writer %d about to write: %s", writer_id, message.get("date"))
                self.write_step(message)
                LOG.info("Writer %d finished processing: %s", writer_id, message.get("date"))
            except mp.queues.Empty:
                LOG.info("Writer %d queue empty – shutting down", writer_id)
                break
            except Exception as e:
                LOG.error("Writer %d encountered error: %s\n%s", writer_id, e, traceback.format_exc())
                break

        LOG.info("Writer %d shutting down", writer_id)

    def _split_state(self, state: State, num_chunks: int) -> list[State]:
        """Split a state into *num_chunks* chunks along the fields dimension.

        All non-field entries are replicated across chunks.

        Parameters
        ----------
        state : State
            The state to split.
        num_chunks : int
            Number of chunks to produce.

        Returns
        -------
        list[State]
            A list of state chunks.
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

    def _terminate_all_writers(self) -> None:
        """Terminate all writer processes gracefully."""
        if self.num_writers == 0:
            return

        self._writers_running = False
        LOG.info("Initiating writer shutdown...")

        # Signal each writer to terminate
        for i, queue in enumerate(self._queues):
            try:
                queue.put_nowait("TERMINATE")
            except Exception as e:
                LOG.error("Failed to send termination to writer %d: %s", i, e)

        # Wait for processes to finish
        for p in self._processes:
            p.join(timeout=30)
            if p.is_alive():
                LOG.warning("Writer process %s did not exit in time – terminating", p.name)
                p.terminate()

        self._stop_event.set()
        LOG.info("All writer processes terminated and cleaned up.")

    @abstractmethod
    def per_writer_init(self, writer_id: int) -> None:
        """Hook for output-specific per-writer initialisation.

        Override in subclasses to perform setup in each writer process,
        e.g. appending ``_{writer_id}`` to an output file path.

        Parameters
        ----------
        writer_id : int
            Unique identifier for this writer.
        """
        pass

    @staticmethod
    def _detach_tensors(obj):
        """Recursively convert torch tensors to numpy arrays in a state dict.

        This prevents CUDA re-initialization errors in forked writer processes.
        """
        try:
            import torch
        except ImportError:
            return obj

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        if isinstance(obj, dict):
            return {k: Output._detach_tensors(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(Output._detach_tensors(v) for v in obj)
        return obj


class ForwardOutput(Output):
    """Subclass of Output that forwards calls to other outputs.

    Subclass from this class to implement the desired behaviour of `output_frequency`
    which should only apply to leaves.

    """

    def __init__(
        self,
        context: "Context",
        metadata: "Metadata",
        output: Output | Any,
        variables: list[str] | None = None,
        post_processors: list[ProcessorConfig] | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
    ):
        """Initialise the ForwardOutput object.

        Parameters
        ----------
        context : Context
            The context in which the output operates.
        output : Output | Any
            The output configuration dictionary or an Output instance.
        variables : list, optional
            The list of variables, by default None.
        post_processors : Optional[List[ProcessorConfig]], default None
            Post-processors to apply to the input
        output_frequency : Optional[int], optional
            The frequency at which to output states, by default None.
        write_initial_state : Optional[bool], optional
            Whether to write the initial state, by default None.
        """

        from anemoi.inference.outputs import create_output

        super().__init__(
            context,
            metadata,
            variables=variables,
            post_processors=post_processors,
            output_frequency=None,
            write_initial_state=write_initial_state,
        )
        if not isinstance(output, Output):
            output = create_output(context, output, self.metadata)
        self.output = output

        if self.context.output_frequency is not None:
            LOG.warning("output_frequency is ignored for '%s'", self.__class__.__name__)

    @cached_property
    def output_frequency(self) -> datetime.timedelta | None:
        """Get the output frequency."""
        return None

    def modify_state(self, state: State) -> State:
        """Modify the state before writing.

        Parameters
        ----------
        state : State
            The state to modify.

        Returns
        -------
        State
            The modified state.
        """
        return state

    def open(self, state) -> None:
        """Open the output for writing.
        Parameters
        ----------
        state : State
            The initial state.
        """
        self.output.open(self.modify_state(state))

    def close(self) -> None:
        """Close the output."""

        self.output.close()

    def write_initial_state(self, state: State) -> None:
        """Write the initial step of the state.

        Parameters
        ----------
        state : State
            The state dictionary.
        """
        state.setdefault("step", datetime.timedelta(0))

        self.output.write_initial_state(self.modify_state(state))

    def write_step(self, state: State) -> None:
        """Write a step of the state.

        Parameters
        ----------
        state : State
            The state to write.
        """
        self.output.write_state(self.modify_state(state))

    def print_summary(self, depth: int = 0) -> None:
        """Print a summary of the output.

        Parameters
        ----------
        depth : int, optional
            The depth of the summary, by default 0.
        """
        super().print_summary(depth)
        self.output.print_summary(depth + 1)
