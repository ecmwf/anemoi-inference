# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

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
        task.run(self)

    def wait(self):
        self.comm.barrier()

    def send_array(self, sender, tensor, target, tag):
        assert sender.name != target.name, f"Cannot send to self {sender}"
        LOG.info(f"{sender}: sending to {target} {tag}")
        self.comm.Send(tensor, dest=self.ranks[target.name], tag=tag)
        LOG.info(f"{sender}: sent to {target} {tag}")

    def receive_array(self, receiver, tensor, source, tag):
        assert receiver.name != source.name, f"Cannot receive from self {receiver}"
        LOG.info(f"{receiver}: receiving from {source} {tag}")
        self.comm.Recv(tensor, source=self.ranks[source.name], tag=tag)
        LOG.info(f"{receiver}: received from {source} {tag}")
