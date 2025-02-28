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
from typing import Dict

from anemoi.utils.logs import set_logging_name

from ..transport import Transport
from . import transport_registry

LOG = logging.getLogger(__name__)


@transport_registry.register("mpi")
class MPITransport(Transport):
    def __init__(self, couplings: Any, tasks: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        from mpi4py import MPI

        super().__init__(couplings, tasks)
        self.comm: MPI.Comm = MPI.COMM_WORLD
        self.rank: int = self.comm.Get_rank()
        self.size: int = self.comm.Get_size()

        assert (
            len(tasks) == self.size
        ), f"Number of tasks ({len(tasks)}) must match number of MPI processes ({self.size})"

    def start(self) -> None:

        tasks = list(self.tasks.values())
        self.ranks = {task.name: i for i, task in enumerate(tasks)}

        # Pick only one task per rank
        task = tasks[self.rank]
        set_logging_name(task.name)

        task.run(self)

    def wait(self) -> None:
        self.comm.barrier()

    def send(self, sender: Any, target: Any, state: Any, tag: int) -> None:
        # TODO: use Send() to send numpy arrays, if faster
        self.comm.send(state, dest=self.ranks[target.name], tag=tag)

    def receive(self, receiver: Any, source: Any, tag: int) -> Any:
        return self.comm.recv(source=self.ranks[source.name], tag=tag)
