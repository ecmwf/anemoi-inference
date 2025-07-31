# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
import pickle
import struct
from typing import Any

from anemoi.utils.logs import enable_logging_name
from anemoi.utils.logs import set_logging_name

from anemoi.inference.task import Task
from anemoi.inference.types import State

from ..transport import Transport
from . import transport_registry

LOG = logging.getLogger(__name__)


@transport_registry.register("processes")
class ProcessesTransport(Transport):
    """Transport implementation using processes."""

    def __init__(self, couplings: Any, tasks: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """Initialize the ProcessesTransport.

        Parameters
        ----------
        couplings : Any
            The couplings for the transport.
        tasks : Dict[str, Any]
            The tasks to be executed.
        """
        super().__init__(couplings, tasks)
        self.children: dict[int, str] = {}
        self.pipes: dict[tuple[str, str], tuple[int, int]] = {}
        enable_logging_name("main")

    def child_process(self, task: Any) -> int:
        """Run the task in a child process.

        Parameters
        ----------
        task : Any
            The task to be run.

        Returns
        -------
        int
            The exit status of the child process.
        """
        set_logging_name(task.name)

        # Close all the pipes that are not needed
        for (task1, task2), (read_fd, write_fd) in self.pipes.items():
            if task.name not in (task1, task2):
                os.close(read_fd)
                os.close(write_fd)

        try:
            task.run(self)
        except Exception as e:
            LOG.exception(e)
            return 1
        return 0

    def start(self) -> None:
        """Start the transport by forking processes for each task."""

        # Many to many pipes. May not scale well
        # We can use the couplings to reduce the number of pipes
        self.pipes = {}
        for task1 in self.tasks:
            for task2 in self.tasks:
                if task1 != task2:
                    read_fd, write_fd = os.pipe()
                    os.set_inheritable(read_fd, True)
                    os.set_inheritable(write_fd, True)
                    self.pipes[(task1, task2)] = (read_fd, write_fd)

        for name, task in self.tasks.items():
            pid = os.fork()
            if pid == 0:
                os._exit(self.child_process(task))
            else:
                self.children[pid] = name

        # We need to close the pipes in the parent process
        for read_fd, write_fd in self.pipes.values():
            os.close(read_fd)
            os.close(write_fd)

    def wait(self) -> None:
        """Wait for all child processes to complete and handle any errors."""
        while self.children:
            (pid, status) = os.wait()
            LOG.info(f"Child process {pid} ({self.children[pid]}) exited with status {status}")
            del self.children[pid]

            if status != 0:
                for pid in self.children:
                    os.kill(pid, 15)

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
        # TODO: something more efficient than pickle
        _, write_fd = self.pipes[(sender.name, target.name)]
        pickle_data = pickle.dumps(state)

        os.write(write_fd, struct.pack("!Q", tag))
        os.write(write_fd, struct.pack("!Q", len(pickle_data)))
        os.write(write_fd, pickle_data)

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
        read_fd, _ = self.pipes[(source.name, receiver.name)]

        recieved_tag = struct.unpack("!Q", os.read(read_fd, 8))[0]
        assert recieved_tag == tag, (recieved_tag, tag)

        size = struct.unpack("!Q", os.read(read_fd, 8))[0]
        data = os.read(read_fd, size)
        state = pickle.loads(data)
        if isinstance(state, Exception):
            raise state

        return state
