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
from typing import Any
from typing import Callable

from anemoi.inference.clusters import cluster_registry
from anemoi.inference.clusters.mapping import EnvMapping
from anemoi.inference.clusters.mapping import MappingCluster
from anemoi.inference.clusters.spawner import ComputeSpawner

LOG = logging.getLogger(__name__)

MANUAL_ENV_MARKER = "ANEMOI_INFERENCE_MANUAL_CLUSTER"

MANUAL_ENV_MAPPING = EnvMapping(
    local_rank="ANEMOI_INFERENCE_MANUAL_LOCAL_RANK",
    global_rank="ANEMOI_INFERENCE_MANUAL_RANK",
    world_size="ANEMOI_INFERENCE_MANUAL_WORLD_SIZE",
    master_addr="ANEMOI_INFERENCE_MANUAL_MASTER_ADDR",
    master_port="ANEMOI_INFERENCE_MANUAL_MASTER_PORT",
    init_method="tcp://{master_addr}:{master_port}",
)


def _execute_with_env(
    fn: Callable, *args: tuple[Any, ...], env_kwargs: dict[str, Any], **kwargs: dict[str, Any]
) -> None:
    """Execute the function with environment variables set."""
    saved_env = os.environ.copy()

    for key, env_var in env_kwargs.items():
        os.environ[key] = str(env_var)
    return_var = fn(*args, **kwargs)

    os.environ.update(saved_env)
    return return_var


@cluster_registry.register("manual")  # type: ignore
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

    def __new__(cls, *args: Any, **kwargs: Any):
        if os.environ.get(MANUAL_ENV_MARKER) == "active":
            return ManualClient()
        return super().__new__(cls)

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

    def spawn(self, fn: Any, *args: Any) -> None:
        os.environ[MANUAL_ENV_MARKER] = "active"
        import torch.multiprocessing as mp

        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            LOG.warning("Multiprocessing start method has already been set.")

        port = self._create_port()
        mapping = {
            MANUAL_ENV_MAPPING.world_size: str(self._world_size),
            MANUAL_ENV_MAPPING.master_addr: "localhost",
            MANUAL_ENV_MAPPING.master_port: str(port),
            MANUAL_ENV_MARKER: "active",
        }

        for pid in range(self._world_size):
            pid_mapping = {
                **mapping,
                MANUAL_ENV_MAPPING.global_rank: str(pid),
                MANUAL_ENV_MAPPING.local_rank: str(pid),
            }

            process = mp.Process(target=_execute_with_env, args=(fn, *args), kwargs={"env_kwargs": pid_mapping})
            process.start()
            self._spawned_processes.append(process)

        # Ensure all spawned processes complete execution
        for process in self._spawned_processes:
            process.join()

    def teardown(self) -> None:
        """Tear down the cluster environment and join spawned processes."""
        # Join all spawned processes to ensure clean shutdown
        for process in self._spawned_processes:
            if process.is_alive():
                process.join(timeout=10)
                if process.is_alive():
                    LOG.warning(f"Process {process.pid} did not terminate, forcing...")
                    process.terminate()
                    process.join(timeout=5)


class ManualClient(MappingCluster):  # type: ignore
    def __init__(self) -> None:
        """Initialise the ManualClient."""
        super().__init__(mapping=MANUAL_ENV_MAPPING)

    @classmethod
    def used(cls) -> bool:
        return bool(os.environ.get(MANUAL_ENV_MARKER))
