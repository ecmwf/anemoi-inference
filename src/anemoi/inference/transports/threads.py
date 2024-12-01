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

from ..transport import Transport
from . import transport_registry

LOG = logging.getLogger(__name__)


class TaskWrapper:
    """_summary_"""

    def __init__(self, task):
        self.task = task
        self.queue = queue.Queue(maxsize=1)
        self.error = None
        self.name = task.name

    def run(self, transport):
        try:
            self.task.run(transport)
        except Exception as e:
            LOG.exception(e)
            self.error = e

    def __repr__(self) -> str:
        return repr(self.task)


@transport_registry.register("threads")
class ThreadsTransport(Transport):
    """_summary_"""

    def __init__(self, couplings, rpcs, tasks, *args, **kwargs):
        super().__init__(couplings, rpcs, tasks)
        self.threads = {}
        self.lock = threading.Lock()
        self.backlogs = {name: {} for name in tasks}

    def start(self):
        self.wrapped_tasks = {name: TaskWrapper(task) for name, task in self.tasks.items()}

        for name, wrapped_task in self.wrapped_tasks.items():
            self.threads[name] = threading.Thread(target=wrapped_task.run, args=(self,))
            self.threads[name].start()

    def wait(self):
        for name, thread in self.threads.items():
            thread.join()
            LOG.info(f"Thread `{name}` finished")

        for name, wrapped_task in self.wrapped_tasks.items():
            if wrapped_task.error:
                raise wrapped_task.error

    def send_array(self, sender, tensor, target, tag):
        assert sender.name != target.name, f"Cannot send to self {sender}"
        LOG.info(f"{sender}: sending to {target} {tag}")
        self.wrapped_tasks[target.name].queue.put((sender.name, tensor, tag))
        LOG.info(f"{sender}: sent to {target} {tag}")

    def receive_array(self, receiver, tensor, source, tag):
        assert receiver.name != source.name, f"Cannot receive from self {receiver}"
        LOG.info(f"{receiver}: receiving from {source} {tag} (backlog: {len(self.backlogs[receiver.name])})")

        if (source.name, tag) in self.backlogs[receiver.name]:
            with self.lock:
                data = self.backlogs[receiver.name].pop((source.name, tag))
            tensor[:] = data
            LOG.info(f"{receiver}: received from {source} {tag} (from backlog)")
            return

        while True:
            (sender, data, tag) = self.wrapped_tasks[receiver.name].queue.get()
            if sender != source.name or tag != tag:
                with self.lock:
                    self.backlogs[receiver.name][(sender, tag)] = data
                continue

            tensor[:] = data
            LOG.info(f"{receiver}: received from {source} {tag}")
            break
