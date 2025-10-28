# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
import os
from functools import cached_property
from typing import Any

from anemoi.inference.clusters import Cluster
from anemoi.inference.clusters import cluster_registry
from anemoi.inference.context import Context
from anemoi.inference.decorators import main_argument

LOG = logging.getLogger(__name__)


@cluster_registry.register("manual")  # type: ignore
@main_argument("world_size")
class ManualCluster(Cluster):
    """Manual cluster that uses user-defined world size for distributed setup.

    Example usage
    -------------
    In the config
    ```yaml
    cluster:
        manual:
            world_size: 4
    ```

    ```python
    from anemoi.inference.clusters.manual import ManualCluster
    cluster = ManualCluster(context, world_size=4)
    ```
    """

    _master_addr: str
    _master_port: int

    def __init__(self, context: Context, *, world_size: int, pid: int = 0) -> None:
        super().__init__(context)
        self._world_size = world_size
        self.pid = pid
        self._spawned_processes = []
        if self.world_size <= 0:
            raise ValueError(
                "Error. 'world_size' must be greater then 1 to use parallel inference, set `cluster.manual.world_size`."
            )

    @property
    def global_rank(self) -> int:
        """Return the rank of the current process."""
        return self.pid

    @property
    def local_rank(self) -> int:
        """Return the rank of the current process."""
        return self.pid

    @property
    def world_size(self) -> int:
        """Return the total number of processes in the cluster."""
        return self._world_size

    @property
    def master_addr(self) -> str:
        return "localhost"

    @cached_property
    def master_port(self) -> int:  # type: ignore
        import hashlib

        node_name = os.uname().nodename.encode()  # Convert to bytes
        hash_val = int(hashlib.md5(node_name).hexdigest(), 16)  # Convert hash to int
        master_port = 10000 + (hash_val % 9999)
        return master_port

    def spawn(self, fn: Any, *args: Any) -> None:
        if self.pid != 0:
            return  # only the main process should spawn others

        import torch.multiprocessing as mp

        mp.set_start_method("spawn")

        for pid in range(1, self.world_size):
            process = mp.Process(target=fn, args=args, kwargs={"pid": pid})
            process.start()
            self._spawned_processes.append(process)

    def teardown(self) -> None:
        """Tear down the cluster environment and join spawned processes."""
        super().teardown()

        # Join all spawned processes to ensure clean shutdown
        for process in self._spawned_processes:
            if process.is_alive():
                process.join(timeout=10)
                if process.is_alive():
                    LOG.warning(f"Process {process.pid} did not terminate, forcing...")
                    process.terminate()
                    process.join(timeout=5)

    @classmethod
    def used(cls) -> bool:
        return False
