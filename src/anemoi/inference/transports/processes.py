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
import select
import struct

import numpy as np

from ..transport import Transport
from . import transport_registry

LOG = logging.getLogger(__name__)


@transport_registry.register("processes")
class ProcessesTransport(Transport):
    """_summary_"""

    def __init__(self, couplings, rpcs, tasks, *args, **kwargs):
        super().__init__(couplings, rpcs, tasks)
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
                self.children[name] = pid

        # We need to close the pipes in the parent process
        for read_fd, write_fd in self.pipes.values():
            os.close(read_fd)
            os.close(write_fd)

    def wait(self):
        for name, pid in self.children.items():
            if os.waitpid(pid, 0) != (pid, 0):
                raise RuntimeError("Child process failed %s %s" % (name, pid))

    def send_array(self, sender, tensor, target, tag):
        assert sender.name != target.name, f"Cannot send to self {sender}"
        _, write_fd = self.pipes[(sender.name, target.name)]

        os.write(write_fd, "a".encode())  # a for array

        header = np.array([tag, tensor.size * tensor.itemsize], dtype=np.uint64)

        os.write(write_fd, header.tobytes())
        os.write(write_fd, tensor.tobytes())

    def receive_array(self, receiver, tensor, source, tag):
        assert receiver.name != source.name, f"Cannot receive from self {receiver}"
        tag = np.uint64(tag)

        read_fd, _ = self.pipes[(source.name, receiver.name)]

        code = os.read(read_fd, 1).decode()
        assert code == "a", f"Expected array got {code}"

        # Read the data
        header = os.read(read_fd, np.dtype(np.uint64).itemsize * 2)
        header = np.frombuffer(header, dtype=np.uint64)
        assert tag == header[0]
        size = header[1]
        tensor[:] = np.ndarray(buffer=os.read(read_fd, size), dtype=tensor.dtype, shape=tensor.shape)
        LOG.info(f"{receiver}: received from {source} {tag}")

    def rpc(self, sender, proc, *args, **kwargs):

        target = self.rpcs[proc]

        assert sender.name != target, f"Cannot send to self {sender}"
        _, write_fd = self.pipes[(sender.name, target)]
        read_fd, _ = self.pipes[(target, sender.name)]

        LOG.info(f"{sender}: sending rpc {proc} to {target} {read_fd} {write_fd}")

        os.write(write_fd, "r".encode())
        data = pickle.dumps((proc, args, kwargs))
        os.write(write_fd, struct.pack("!I", len(data)))
        os.write(write_fd, data)

        code = os.read(read_fd, 1).decode()
        assert code == "r", f"Expected array got {code}"
        size = struct.unpack("!I", os.read(read_fd, 4))[0]
        data = os.read(read_fd, size)
        result = pickle.loads(data)
        if isinstance(result, Exception):
            raise result
        return result

    def dispatch(self, task, dispatcher):
        LOG.info(f"{task}: waiting for messages {self.pipes} {task.name}")
        while True:
            fds = [fd[0] for (peers, fd) in self.pipes.items() if task.name == peers[1]]
            remotes = {fd[0]: peers[0] for (peers, fd) in self.pipes.items() if task.name == peers[1]}

            if not fds:
                LOG.info(f"{task}: no more messages")
                break

            LOG.info(f"{task}: waiting on {fds}")
            read_fds, _, _ = select.select(fds, [], [])
            LOG.info(f"{task}: got message {read_fds}")

            for read_fd in read_fds:

                LOG.info(f"{task}: reading from {read_fd}, remote is {remotes[read_fd]}")

                code = os.read(read_fd, 1).decode()
                assert code == "r", f"Expected array got {code}"
                size = struct.unpack("!I", os.read(read_fd, 4))[0]
                data = os.read(read_fd, size)
                (proc, args, kwargs) = pickle.loads(data)

                LOG.info(f"{task}: received rpc {proc} {args} {kwargs}")

                try:
                    result = dispatcher[proc](*args, **kwargs)
                except Exception as e:
                    LOG.exception(e)
                    result = e

                _, write_fd = self.pipes[(task.name, remotes[read_fd])]
                os.write(write_fd, "r".encode())
                data = pickle.dumps(result)
                os.write(write_fd, struct.pack("!I", len(data)))
                os.write(write_fd, data)
