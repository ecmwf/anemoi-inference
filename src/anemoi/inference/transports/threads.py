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

from anemoi.utils.logs import set_logging_name

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
        set_logging_name(self.task.name)
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

    def __init__(self, couplings, tasks, *args, **kwargs):
        super().__init__(couplings, tasks)
        self.threads = {}
        self.lock = threading.Lock()
        self.backlogs = {name: {} for name in tasks}

    def start(self):
        self.wrapped_tasks = {name: TaskWrapper(task) for name, task in self.tasks.items()}

        for name, wrapped_task in self.wrapped_tasks.items():
            self.threads[name] = threading.Thread(target=wrapped_task.run, args=(self,))
            self.threads[name].start()

    def wait(self):
        # TODO: wait for all threads, and kill remaining threads if any of them failed
        for name, thread in self.threads.items():
            thread.join()
            LOG.info(f"Thread `{name}` finished")

        for name, wrapped_task in self.wrapped_tasks.items():
            if wrapped_task.error:
                raise wrapped_task.error

    def send(self, sender, target, state, tag):
        self.wrapped_tasks[target.name].queue.put((sender.name, tag, state.copy()))

    def receive(self, receiver, source, tag):
        LOG.info(f"{receiver}: receiving from {source} [{tag}] (backlog: {len(self.backlogs[receiver.name])})")

        if (source.name, tag) in self.backlogs[receiver.name]:
            with self.lock:
                return self.backlogs[receiver.name].pop((source.name, tag))

        while True:
            (sender, tag, state) = self.wrapped_tasks[receiver.name].queue.get()
            if sender != source.name or tag != tag:
                with self.lock:
                    self.backlogs[receiver.name][(sender, tag)] = state
                continue

            return state
