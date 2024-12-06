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

from anemoi.utils.logs import enable_logging_name
from anemoi.utils.logs import set_logging_name

from ..transport import Transport
from . import transport_registry

LOG = logging.getLogger(__name__)


@transport_registry.register("processes")
class ProcessesTransport(Transport):
    """_summary_"""

    def __init__(self, couplings, rpcs, tasks, *args, **kwargs):
        super().__init__(couplings, rpcs, tasks)
        self.children = {}
        enable_logging_name("main")

    def child_process(self, task):
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
                self.children[pid] = name

        # We need to close the pipes in the parent process
        for read_fd, write_fd in self.pipes.values():
            os.close(read_fd)
            os.close(write_fd)

    def wait(self):
        while self.children:
            (pid, status) = os.wait()
            LOG.info(f"Child process {pid} ({self.children[pid]}) exited with status {status}")
            del self.children[pid]

            if status != 0:
                for pid in self.children:
                    os.kill(pid, 15)

    def send_state(self, sender, target, *, input_state, variables, constants):

        assert isinstance(input_state, dict)

        assert sender.name != target.name, f"Cannot send to self {sender}"
        _, write_fd = self.pipes[(sender.name, target.name)]

        fields = input_state["fields"]

        LOG.info(f"{sender}: sending to {target} {variables} {input_state['date']}")

        fields = {v: fields[v] for v in variables if v in fields}

        for v in variables:
            if v not in fields:
                # Check in the constants
                if v in constants:
                    LOG.warning(f"{sender}: {v} not in fields, using the value from constants")
                    fields[v] = constants[v]
                else:
                    raise ValueError(f"{sender}: Variable {v} not in fields or constants")

        for f, v in fields.items():
            assert len(v.shape) == 1, f"Expected  got {v.shape}"

        state = input_state.copy()
        state["fields"] = fields

        # Don't send unnecessary data
        state["latitudes"] = None
        state["longitudes"] = None
        for s in list(state.keys()):
            if s.startswith("_"):
                del state[s]

        # TODO: something more efficient than pickle

        pickle_data = pickle.dumps(state)

        os.write(write_fd, struct.pack("!Q", len(pickle_data)))
        os.write(write_fd, pickle_data)

    def receive_state(self, receiver, source, *, output_state, variables):

        assert receiver.name != source.name, f"Cannot receive from self {receiver}"

        read_fd, _ = self.pipes[(source.name, receiver.name)]

        size = struct.unpack("!Q", os.read(read_fd, 8))[0]
        data = os.read(read_fd, size)
        state = pickle.loads(data)
        if isinstance(state, Exception):
            raise state

        assert isinstance(state, dict)
        assert "fields" in state
        assert isinstance(state["fields"], dict), f"Expected dict got {type(state['fields'])}"

        output_state.setdefault("fields", {})

        fields_in = state["fields"]
        fields_out = output_state["fields"]

        for v in variables:
            if v in fields_out:
                raise ValueError(f"Variable {v} already in output state")

            if v not in fields_in:
                raise ValueError(f"Variable {v} not in input state")

            fields_out[v] = fields_in[v]

            assert len(fields_out[v].shape) == 1, f"Expected  got {fields_out[v].shape}"
