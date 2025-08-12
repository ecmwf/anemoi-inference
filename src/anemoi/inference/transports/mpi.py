# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

from anemoi.utils.logs import set_logging_name

from anemoi.inference.config import Configuration
from anemoi.inference.task import Task
from anemoi.inference.types import State

from ..transport import Transport
from . import transport_registry

LOG = logging.getLogger(__name__)


@transport_registry.register("mpi")
class MPITransport(Transport):
    """Transport implementation using MPI."""

    def __init__(self, couplings: Configuration, tasks: dict[str, Task], *args: Any, **kwargs: Any) -> None:
        """Initialize the MPITransport.

        Parameters
        ----------
        couplings : Configuration
            The couplings for the transport.
        tasks : Dict[str, Any]
            The tasks to be executed.
        """
        from mpi4py import MPI

        super().__init__(couplings, tasks)
        self.comm: MPI.Comm = MPI.COMM_WORLD
        self.rank: int = self.comm.Get_rank()
        self.size: int = self.comm.Get_size()

        assert (
            len(tasks) == self.size
        ), f"Number of tasks ({len(tasks)}) must match number of MPI processes ({self.size})"

    def start(self) -> None:
        """Start the transport by initializing MPI tasks."""

        tasks = list(self.tasks.values())
        self.ranks = {task.name: i for i, task in enumerate(tasks)}

        # Pick only one task per rank
        task = tasks[self.rank]
        set_logging_name(task.name)

        task.run(self)

    def wait(self) -> None:
        """Wait for all MPI tasks to complete."""
        self.comm.barrier()

    def send(self, sender: Task, target: Task, state: State, tag: int) -> None:
        """Send a state from the sender to the target.

        Parameters
        ----------
        sender : Task
            The task sending the state.
        target : Task
            The task receiving the state.
        state : State
            The state to be sent.
        tag : int
            The tag associated with the state.
        """
        self.comm.send(state, dest=self.ranks[target.name], tag=tag)

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
        return self.comm.recv(source=self.ranks[source.name], tag=tag)
