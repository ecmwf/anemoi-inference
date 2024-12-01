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

import numpy as np

from ..transport import Transport
from . import transport_registry

LOG = logging.getLogger(__name__)


@transport_registry.register("processes")
class ProcessesTransport(Transport):
    """_summary_"""

    def __init__(self, couplings, tasks, *args, **kwargs):
        super().__init__(couplings, tasks)
        self.children = {}

    def child_process(self, task):

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

    def start(self):

        # Many to many pipes. May not scale well
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
                self.children[name] = pid

        # We need to close the pipes in the parent process
        for read_fd, write_fd in self.pipes.values():
            os.close(read_fd)
            os.close(write_fd)

    def wait(self):
        for name, pid in self.children.items():
            if os.waitpid(pid, 0) != (pid, 0):
                raise RuntimeError("Child process failed %s %s" % (name, pid))

    def send(self, sender, tensor, target, tag):
        assert sender.name != target.name, f"Cannot send to self {sender}"
        # Find the pipe
        _, write_fd = self.pipes[(sender.name, target.name)]
        # Write the data

        header = np.array([tag, tensor.size * tensor.itemsize], dtype=np.uint64)

        os.write(write_fd, header.tobytes())
        os.write(write_fd, tensor.tobytes())

    def receive(self, receiver, tensor, source, tag):
        assert receiver.name != source.name, f"Cannot receive from self {receiver}"
        tag = np.uint64(tag)
        # Find
        read_fd, _ = self.pipes[(source.name, receiver.name)]
        # Read the data
        header = os.read(read_fd, np.dtype(np.uint64).itemsize * 2)
        header = np.frombuffer(header, dtype=np.uint64)
        assert tag == header[0]
        size = header[1]
        tensor[:] = np.ndarray(buffer=os.read(read_fd, size), dtype=tensor.dtype, shape=tensor.shape)
        LOG.info(f"{receiver}: received from {source} {tag}")
