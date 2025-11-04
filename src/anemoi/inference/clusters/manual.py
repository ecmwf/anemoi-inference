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

from anemoi.inference.clusters import cluster_registry
from anemoi.inference.clusters.client import ComputeClientFactory
from anemoi.inference.clusters.spawner import SPAWN_FUNCTION
from anemoi.inference.clusters.spawner import ComputeSpawner
from anemoi.inference.config import Configuration

LOG = logging.getLogger(__name__)


@cluster_registry.register("manual")
class ManualSpawner(ComputeSpawner):
    """Manual cluster that uses user-defined world size for distributed setup.

    Example usage
    -------------
    In the config
    ```yaml
    cluster:
        manual:
            world_size: 4
            port: 12345
    ```
    """

    def __init__(self, world_size: int, port: int | None = None) -> None:
        if world_size < 1:
            raise ValueError("world_size must be at least 1.")
        self._world_size = world_size
        self._port = port
        self._spawned_processes = []

    @classmethod
    def used(cls) -> bool:
        return False

    def _create_port(self) -> int:
        """Create a unique port based on the node name."""
        if self._port is not None:
            return self._port

        import hashlib

        node_name = os.uname().nodename.encode()  # Convert to bytes
        hash_val = int(hashlib.md5(node_name).hexdigest(), 16)  # Convert hash to int
        master_port = 10000 + (hash_val % 9999)
        return master_port

    def spawn(self, fn: SPAWN_FUNCTION, config: "Configuration") -> None:
        import torch.multiprocessing as mp

        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            LOG.warning("Multiprocessing start method has already been set.")

        port = self._create_port()

        for pid in range(self._world_size):
            factory = ManualClient(
                world_size=self._world_size, local_rank=pid, global_rank=pid, master_addr="localhost", master_port=port
            )
            process = mp.Process(target=fn, args=(config, factory))
            process.start()
            self._spawned_processes.append(process)

        # Ensure all spawned processes complete execution
        for process in self._spawned_processes:
            process.join()

    def teardown(self) -> None:
        """Tear down the cluster environment and join spawned processes."""
        # Join all spawned processes to ensure clean shutdown
        for process in self._spawned_processes:
            if not process.is_alive():
                continue

            process.terminate()
            process.join(1)
            if process.exitcode is None:
                LOG.debug(f"Kill hung process - PID: {process.pid}")
                process.kill()


class ManualClient(ComputeClientFactory):
    def __init__(self, world_size: int, local_rank: int, global_rank: int, master_addr: str, master_port: int) -> None:
        """Initialise the ManualClient."""
        self._world_size = world_size
        self._local_rank = local_rank
        self._global_rank = global_rank
        self._master_addr = master_addr
        self._master_port = master_port

    @classmethod
    def used(cls) -> bool:
        return True

    @property
    def world_size(self) -> int:
        """Return the total number of processes in the cluster."""
        return self._world_size

    @property
    def global_rank(self) -> int:
        """Return the rank of the current process."""
        return self._global_rank

    @property
    def local_rank(self) -> int:
        """Return the rank of the current process."""
        return self._local_rank

    @property
    def master_addr(self) -> str:
        """Return the master address."""
        return self._master_addr

    @property
    def master_port(self) -> int:
        """Return the master port."""
        return self._master_port
