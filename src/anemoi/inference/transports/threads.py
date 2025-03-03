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
from typing import Any
from typing import Dict

from anemoi.utils.logs import set_logging_name

from ..transport import Transport
from . import transport_registry

LOG = logging.getLogger(__name__)


class TaskWrapper:

    def __init__(self, task: Any) -> None:
        self.task: Any = task
        self.queue: queue.Queue = queue.Queue(maxsize=1)
        self.error: Exception | None = None
        self.name: str = task.name

    def run(self, transport: "ThreadsTransport") -> None:
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

    def __init__(self, couplings: Any, tasks: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(couplings, tasks)
        self.threads: Dict[str, threading.Thread] = {}
        self.lock: threading.Lock = threading.Lock()
        self.backlogs: Dict[str, Dict[tuple, Any]] = {name: {} for name in tasks}

    def start(self) -> None:
        self.wrapped_tasks = {name: TaskWrapper(task) for name, task in self.tasks.items()}

        for name, wrapped_task in self.wrapped_tasks.items():
            self.threads[name] = threading.Thread(target=wrapped_task.run, args=(self,))
            self.threads[name].start()

    def wait(self) -> None:
        # TODO: wait for all threads, and kill remaining threads if any of them failed
        for name, thread in self.threads.items():
            thread.join()
            LOG.info(f"Thread `{name}` finished")

        for name, wrapped_task in self.wrapped_tasks.items():
            if wrapped_task.error:
                raise wrapped_task.error

    def send(self, sender: Any, target: Any, state: Any, tag: int) -> None:
        self.wrapped_tasks[target.name].queue.put((sender.name, tag, state.copy()))

    def receive(self, receiver: Any, source: Any, tag: int) -> Any:
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
