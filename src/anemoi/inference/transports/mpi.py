# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from anemoi.utils.logs import set_logging_name

from ..transport import Transport
from . import transport_registry

LOG = logging.getLogger(__name__)


@transport_registry.register("mpi")
class MPITransport(Transport):
    """_summary_"""

    def __init__(self, couplings, rpcs, tasks, *args, **kwargs):
        from mpi4py import MPI

        super().__init__(couplings, rpcs, tasks)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        assert (
            len(tasks) == self.size
        ), f"Number of tasks ({len(tasks)}) must match number of MPI processes ({self.size})"

    def start(self):

        tasks = list(self.tasks.values())
        self.ranks = {task.name: i for i, task in enumerate(tasks)}

        # Pick only one task per rank
        task = tasks[self.rank]
        set_logging_name(task.name)

        task.run(self)

    def wait(self):
        self.comm.barrier()

    def send(self, sender, target, state):
        # TODO: use a tag; use Send() to send numpy arrays, if faster
        tag = 0
        self.comm.send(state, dest=self.ranks[target.name], tag=tag)

    def receive(self, receiver, source):
        tag = 0
        return self.comm.recv(source=self.ranks[source.name], tag=tag)
