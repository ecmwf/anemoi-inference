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
from abc import ABC
from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING
from typing import List, Optional, Callable

import multiprocessing as mp
import math
import os
import itertools, types, pickle
import traceback
#import dill #thought this would make lambdas picklable but no look

from anemoi.inference.types import State
import pprint
pp = pprint.PrettyPrinter(indent=4)

if TYPE_CHECKING:
    from anemoi.inference.context import Context

LOG = logging.getLogger(__name__)

from collections.abc import Mapping, Iterable

def find_unpicklable(obj, path="root"):
    try:
        pickle.dumps(obj)
        return  # It's fine
    except Exception as e:
        print(f"❌ Unpicklable at {path}: {e}")

    # If it's a dict, dive into keys and values
    if isinstance(obj, dict):
        for k, v in obj.items():
            find_unpicklable(k, f"{path}[key={repr(k)}]")
            find_unpicklable(v, f"{path}[{repr(k)}]")
    # If it's a list, tuple, set — dive into elements
    elif isinstance(obj, (list, tuple, set)):
        for i, item in enumerate(obj):
            find_unpicklable(item, f"{path}[{i}]")
    # If it's a custom class, inspect __dict__
    elif hasattr(obj, '__dict__'):
        for attr, val in vars(obj).items():
            find_unpicklable(val, f"{path}.{attr}")

def check_picklability(obj):
    try:
        pickle.dumps(obj)
    except Exception as e:
        print(f"Object is not picklable: {e}")
        raise
    
def sanitize_state(state):
    # remove unpicklable fields like lambdas or function refs
    return {
        key: val
        for key, val in state.items()
        if not callable(val)
    }

class Output(ABC):
    """Abstract base class for output mechanisms."""

    def __init__(
        self,
        context: "Context",
        output_frequency: Optional[int] = None,
        write_initial_state: Optional[bool] = None,
        num_writers: Optional[int] = None,
    ):
        """Initialize the Output object.

        Parameters
        ----------
        context : Context
            The context in which the output operates.
        output_frequency : Optional[int], optional
            The frequency at which to output states, by default None.
        write_initial_state : Optional[bool], optional
            Whether to write the initial state, by default None.
        """
        self.context = context
        self.checkpoint = context.checkpoint
        self.reference_date = None

        self._write_step_zero = write_initial_state
        self._output_frequency = output_frequency
        
        self.num_writers = self.context.config.writers_per_device
        if  self.num_writers > 0:
             self.writers_running=False
        
    def _spawn_writers(self):
        """Spawn all writer processes."""
        
        self.processes: List[mp.Process] = []
        self.queues: List[mp.Queue] = []

        self.stop_event = mp.Event()
            
        # Create queues for each writer
        # Each writer will asynchronously write output
        for i in range(self.num_writers):
            self.queues.append(mp.Queue())

        mp.set_start_method('fork', force=True)
        import os
        parent_pid=os.getpid()
        for i in range(self.num_writers):
            process = mp.Process(
                target=self._writer_function,
                args=(i, self.queues[i], self.stop_event),
                name=f"w{i}_for_p{parent_pid}",
            )
            #_pickle.PicklingError: Can't pickle <class 'anemoi.inference.outputs.gribfile.GribFileOutput'>: it's not the same object as anemoi.inference.outputs.gribfile.GribFileOutput
            #I get this error when using the default method, spawn?
            process.start()
            self.processes.append(process)
        
        LOG.info(f"Spawned {self.num_writers} writers per process") # this seems to get called all the time
        
    def _writer_function(self, writer_id: int, queue: mp.Queue, stop_event) -> None:
        """
        Worker function that runs in each writer process.
        Waits for messages from the main process and then writes them asyncronously.
        
        Args:
            worker_id: Unique identifier for this worker
            queue: Queue to receive messages from main process
        """
        #LOG.debug(f"Initialising writer {writer_id}...")
        #self.__init__(context=context)
        LOG.info(f"Writer {writer_id} started and waiting for messages...")
        
        self.per_writer_init(writer_id)
        
        while not stop_event.is_set():
            try:
                # Wait for message from main process
                message = queue.get(timeout=1)  # 1 second timeout
                
                if message == "TERMINATE":
                    LOG.debug(f"Writer {writer_id} received termination signal")
                    break
               
                LOG.info(f"Writer {writer_id} about to write: {message['date']}") 
                self.write_step(message)
                LOG.info(f"Writer {writer_id} finished processing: {message['date']}")
                
            except mp.queues.Empty:
                # No message received, continue waiting
                #LOG.info(f"Writer {writer_id} has an empty queue")
                #break
                continue
            except Exception as e:
                print(traceback.format_exc())
                LOG.error(f"Writer {writer_id} encountered error: {e}")
                break
        
        LOG.info(f"Worker {writer_id} shutting down")
        
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
            self.write_step(state)
            
    def _split_state(self, state: State, num_chunks: int) -> List[State]:
        """ Splits State into 'num_chunks' chunks
        
        State is split along the list in state['fields'].
        All other entries in state are replicated across chunks
        
        """
        if "fields" not in state:
            raise ValueError("State dictionary must contain 'fields' key")
        #if not isinstance(state["fields"], list):
        #    raise ValueError("state['fields'] must be a list")
        
        if num_chunks == 1:
            return [state]
        
        fields = state["fields"]
        fields_per_chunk = math.ceil(len(fields.items()) / num_chunks)
        
        chunks = []
        
        for i in range(num_chunks):
            # Calculate start and end indices for this chunk
            start_idx = i * fields_per_chunk
            end_idx = min(start_idx + fields_per_chunk, len(fields))
            
            # Skip if we've exceeded the fields list
            if start_idx >= len(fields):
                break
            
            # Create new state dictionary for this chunk
            #chunk_state = copy.deepcopy(state)  #wont work bc of pickling issues
            # Replace the fields with the chunk subset

            chunk = state.copy()
            chunk["fields"] = {}
            subset_of_fields=list(state["fields"].items())[start_idx:end_idx]
            # More memory efficient for large dictionaries
            #subset_of_fields = list(itertools.islice(state["fields"].items(), start_idx, end_idx))
            #[rank0]: IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
            for field, values in subset_of_fields:
                chunk["fields"][field] = values #[-1]
            
            chunks.append(chunk)
            
        return chunks
            
        

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
            
        if self.num_writers == 0:
            return self.write_step(state)
        else:
            #use self.num_writers and state['num_fields'] to split state up into chunks
            #spawn 'self.num_writers' processes and have them each write a chunk of 'state'
            #find_unpicklable(state)
            #pp.pprint(state)
            #state
            #I have to spawn the writers at runtime
            #if i spawn at init time, then the output subclass hasnt been initialised,
            # so per_writer_init is too early
            if not self.writers_running:
                self._spawn_writers()
                self.writers_running=True
            state_chunks= self._split_state(state, self.num_writers)
            for i in range(self.num_writers):
                # Find the problematic lambdas
                #check_picklability(state_chunks[i])
                #chunk = sanitize_state(state_chunks[i])
                #self.queues[i].put(chunk)
                #find_unpicklable(state_chunks[i])
                state_chunks[i]['_grib_templates_for_output']={} #check if this is causing the pickle errors
                #i have an error writing grib outptu
                #passing variables through queue.put forces them to be pickled
                # and _grib_templates_for_output cant be pickled bc there's some lambda unctions for earthkit in there
                #self.grib_templates=state['_grib_templates_for_output']
                #pp.pprint(state_chunks[i])
                #exit()
                self.queues[i].put(state_chunks[i])
            #raise ValueError("Multiple writer procs not supported yet")
            
    
            
    def _terminate_all_writers(self):
        """Terminate all writer processes gracefully."""
        if self.num_writers == 0:
            return
        LOG.info("Terminating all worker processes...")
        
        # Send termination signal to all writers
        for queue in self.queues:
            try:
                queue.put("TERMINATE")
            except Exception as e:
                LOG.error(f"Failed to send termination signal: {e}")
        
        # Wait for all processes to finish
        for i, process in enumerate(self.processes):
            process.join(timeout=1)  # Wait up to 5 seconds
            breakpoint()
            if process.is_alive():
                LOG.info(f"Force terminating worker {i}")
                process.terminate()
                process.join()
        
        LOG.info("All writer processes terminated")
        #TODO I could merge all writer procs messages at this point

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
            reduced_state["fields"][field] = values[-1, :]
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
        if self.num_writers > 0:
            self.stop_event.set()
        pass

    @abstractmethod
    def write_step(self, state: State) -> None:
        """Write a step of the state.

        Parameters
        ----------
        state : State
            The state to write.
        """
        pass
    
    def per_writer_init(self, writer_id) -> None:
        """
        Method to allow output-specific initalisation of workers
        e.g. for grib you could append '_{worker_id}' to self.path
        """
        pass

    @cached_property
    def write_step_zero(self) -> bool:
        """Determine whether to write the initial state."""
        if self._write_step_zero is not None:
            return self._write_step_zero

        return self.context.write_initial_state

    @cached_property
    def output_frequency(self) -> Optional[datetime.timedelta]:
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
            "%s%s: output_frequency=%s write_initial_state=%s",
            " " * depth,
            self,
            self.output_frequency,
            self.write_step_zero,
        )


class ForwardOutput(Output):
    """Subclass of Output that forwards calls to other outputs.

    Subclass from this class to implement the desired behaviour of `output_frequency`
    which should only apply to leaves.
    """

    def __init__(
        self,
        context: "Context",
        output: dict,
        output_frequency: Optional[int] = None,
        write_initial_state: Optional[bool] = None,
    ):
        """Initialize the ForwardOutput object.

        Parameters
        ----------
        context : Context
            The context in which the output operates.
        output : dict
            The output configuration dictionary.
        output_frequency : Optional[int], optional
            The frequency at which to output states, by default None.
        write_initial_state : Optional[bool], optional
            Whether to write the initial state, by default None.
        """

        from anemoi.inference.outputs import create_output

        super().__init__(context, output_frequency=None, write_initial_state=write_initial_state)

        self.output = None if output is None else create_output(context, output)

        if self.context.output_frequency is not None:
            LOG.warning("output_frequency is ignored for '%s'", self.__class__.__name__)

    @cached_property
    def output_frequency(self) -> Optional[datetime.timedelta]:
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
