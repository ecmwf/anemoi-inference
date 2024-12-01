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
        self.queue = queue.Queue()
        self.error = None

    def run(self, transport, wrapper):
        try:
            wrapper.task.run(transport)
        except Exception as e:
            LOG.exception(e)
            self.error = e


@transport_registry.register("threads")
class ThreadsTransport(Transport):
    """_summary_"""

    def __init__(self, couplings, *args, **kwargs):
        super().__init__(couplings)
        self.threads = {}
        self.backlog = {}

    def start(self, tasks):
        self.tasks = {name: TaskWrapper(task) for name, task in tasks.items()}

        for name, task in self.tasks.items():
            self.threads[name] = threading.Thread(
                target=task.run,
                args=(
                    self,
                    task,
                ),
            )
            self.threads[name].start()

    def wait(self):
        for name, thread in self.threads.items():
            thread.join()
            LOG.info(f"Thread `{name}` finished")

        for name, task in self.tasks.items():
            if task.error:
                raise task.error

    def send(self, sender, tensor, target, tag):
        LOG.info(f"{sender}: sending to {target} {tag}")
        self.tasks[target.name].queue.put((sender.name, tensor, tag))
        LOG.info(f"{sender}: sent to {target} {tag}")

    def receive(self, receiver, tensor, source, tag):
        LOG.info(f"{receiver}: receiving from {source} {tag}")
        # Check in backlog

        if (source.name, tag) in self.backlog:
            tensor[:] = self.backlog.pop((source.name, tag))
            LOG.info(f"{receiver}: received from {source} {tag} (from backlog)")
            return

        while True:
            (sender, data, tag) = self.tasks[receiver.name].queue.get()
            if sender != source.name or tag != tag:
                self.backlog[(sender, tag)] = data
                continue

            tensor[:] = data
            LOG.info(f"{receiver}: received from {source} {tag}")
            break
