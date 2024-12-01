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

    def __init__(self, couplings, tasks, *args, **kwargs):
        from mpi4py import MPI

        super().__init__(couplings, tasks)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def start(self, tasks):
        assert (
            len(tasks) == self.size
        ), f"Number of tasks ({len(tasks)}) must match number of MPI processes ({self.size})"
        task = list(tasks.values)[self.rank]
        task.run(self)

    def wait(self):
        self.comm.barrier()

    def send(self, sender, tensor, target, tag):
        assert sender.name != target.name, f"Cannot send to self {sender}"
        LOG.info(f"{sender}: sending to {target} {tag}")
        self.tasks[target.name].queue.put((sender.name, tensor, tag))
        LOG.info(f"{sender}: sent to {target} {tag}")

    def receive(self, receiver, tensor, source, tag):
        assert receiver.name != source.name, f"Cannot receive from self {receiver}"
        LOG.info(f"{receiver}: receiving from {source} {tag} (backlog: {len(self.backlogs[receiver.name])})")

        if (source.name, tag) in self.backlogs[receiver.name]:
            with self.lock:
                data = self.backlogs[receiver.name].pop((source.name, tag))
            tensor[:] = data
            LOG.info(f"{receiver}: received from {source} {tag} (from backlog)")
            return

        while True:
            (sender, data, tag) = self.tasks[receiver.name].queue.get()
            if sender != source.name or tag != tag:
                with self.lock:
                    self.backlogs[receiver.name][(sender, tag)] = data
                continue

            tensor[:] = data
            LOG.info(f"{receiver}: received from {source} {tag}")
            break
