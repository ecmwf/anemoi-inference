# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import queue
import threading
from typing import Any
from typing import Dict
from typing import Optional

from anemoi.utils.logs import set_logging_name

from anemoi.inference.task import Task
from anemoi.inference.types import State

from ..transport import Transport
from . import transport_registry

LOG = logging.getLogger(__name__)


class TaskWrapper:
    """Wraps a task to be executed in a thread."""

    def __init__(self, task: Task) -> None:
        """Initialize the TaskWrapper.

        Parameters
        ----------
        task : Task
            The task to be wrapped.
        """
        self.task: Task = task
        self.queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self.error: Optional[Exception] = None
        self.name: str = task.name

    def run(self, transport: "ThreadsTransport") -> None:
        """Run the task within the given transport.

        Parameters
        ----------
        transport : ThreadsTransport
            The transport in which the task is run.
        """
        set_logging_name(self.task.name)
        try:
            self.task.run(transport)
        except Exception as e:
            LOG.exception(e)
            self.error = e

    def __repr__(self) -> str:
        """Return a string representation of the TaskWrapper.

        Returns
        -------
        str
            String representation of the TaskWrapper.
        """
        return repr(self.task)


@transport_registry.register("threads")
class ThreadsTransport(Transport):
    """Transport implementation using threads."""

    def __init__(self, couplings: Any, tasks: Dict[str, Task], *args: Any, **kwargs: Any) -> None:
        """Initialize the ThreadsTransport.

        Parameters
        ----------
        couplings : Any
            The couplings for the transport.
        tasks : Dict[str, Any]
            The tasks to be executed.
        """
        super().__init__(couplings, tasks)
        self.threads: Dict[str, threading.Thread] = {}
        self.lock = threading.Lock()
        self.backlogs: Dict[str, Any] = {name: {} for name in tasks}

    def start(self) -> None:
        """Start the transport by initializing and starting threads for each task."""
        self.wrapped_tasks = {name: TaskWrapper(task) for name, task in self.tasks.items()}

        for name, wrapped_task in self.wrapped_tasks.items():
            self.threads[name] = threading.Thread(target=wrapped_task.run, args=(self,))
            self.threads[name].start()

    def wait(self) -> None:
        """Wait for all threads to complete and handle any errors."""
        # TODO: wait for all threads, and kill remaining threads if any of them failed
        for name, thread in self.threads.items():
            thread.join()
            LOG.info(f"Thread `{name}` finished")

        for name, wrapped_task in self.wrapped_tasks.items():
            if wrapped_task.error:
                raise wrapped_task.error

    def send(self, sender: Task, target: Task, state: State, tag: int) -> None:
        """Send a state from the sender to the target.

        Parameters
        ----------
        sender : Any
            The task sending the state.
        target : Any
            The task receiving the state.
        state : Any
            The state to be sent.
        tag : int
            The tag associated with the state.
        """
        self.wrapped_tasks[target.name].queue.put((sender.name, tag, state.copy()))

    def receive(self, receiver: Task, source: Task, tag: int) -> State:
        """Receive a state from the source to the receiver.

        Parameters
        ----------
        receiver : Any
            The task receiving the state.
        source : Any
            The task sending the state.
        tag : int
            The tag associated with the state.

        Returns
        -------
        Any
            The received state.
        """
        LOG.info(f"{receiver}: receiving from {source} [{tag}] (backlog: {len(self.backlogs[receiver.name])})")

        if (source.name, tag) in self.backlogs[receiver.name]:
            with self.lock:
                return self.backlogs[receiver.name].pop((source.name, tag))

        while True:
            (sender, tag, state) = self.wrapped_tasks[receiver.name].queue.get()
            if sender != source.name or tag != tag:
                with self.lock:
                    self.backlogs[receiver.name][(sender, tag)] = state
                continue

            return state
