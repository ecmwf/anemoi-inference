# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import dataclasses
import logging
import os
from typing import Any

from anemoi.inference.clusters import cluster_registry
from anemoi.inference.clusters.client import ComputeClientFactory

LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class EnvMapping:
    """Dataclass to hold environment variable mappings for cluster configuration.

    Elements can be either strings or lists of strings.
    If a list is provided, the first found environment variable will be used.
    """

    local_rank: str | list[str]
    global_rank: str | list[str]
    world_size: str | list[str]

    master_addr: str | list[str]
    master_port: str | list[str]

    backend: str | None = None
    init_method: str = "env://"

    def get_env(self, key: str, default: Any = None):
        """Get the environment variable value for the given key."""
        mapped_value = getattr(self, key)
        if mapped_value is None:
            return default

        for env_var in (mapped_value if isinstance(mapped_value, list) else [mapped_value]):
            value = os.environ.get(env_var)
            if value is not None:
                return value
        return default


@cluster_registry.register("custom")
class MappingCluster(ComputeClientFactory):
    """Custom cluster that uses user-defined environment variables for distributed setup.

    Example usage
    -------------

    In the config
    ```yaml
    runner:
      parallel:
        cluster:
          custom:
            mapping:
                local_rank: LOCAL_RANK_ENV_VAR
                global_rank: GLOBAL_RANK_ENV_VAR
                world_size: WORLD_SIZE_ENV_VAR
                master_addr: MASTER_ADDR_ENV_VAR
                master_port: MASTER_PORT_ENV_VAR
                init_method: env://
    ```

    ```python
    from anemoi.inference.clusters.mapping import MappingCluster
    cluster = MappingCluster(mapping={
        "local_rank": "LOCAL_RANK_ENV_VAR",
        "global_rank": "GLOBAL_RANK_ENV_VAR",
        "world_size": "WORLD_SIZE_ENV_VAR",
        "master_addr": "MASTER_ADDR_ENV_VAR",
        "master_port": "MASTER_PORT_ENV_VAR",
        "init_method": "env://",
    })
    ```
    """

    def __init__(self, mapping: dict | EnvMapping) -> None:
        """Initalise the MappingCluster

        Parameters
        ----------
        mapping : dict | EnvMapping
            Mapping of environment variables to cluster properties
        """
        self._mapping = EnvMapping(**mapping) if isinstance(mapping, dict) else mapping

    @property
    def init_method(self) -> str:
        """Return the initialisation method string for distributed computing."""
        return self._mapping.init_method.format(master_addr=self.master_addr, master_port=self.master_port)

    @property
    def backend(self) -> str:
        """Return the backend string for distributed computing."""
        return self._mapping.backend or super().backend

    @property
    def world_size(self) -> int:
        """Return the total number of processes in the cluster."""
        return int(self._mapping.get_env("world_size", 1))

    @property
    def global_rank(self) -> int:
        """Return the rank of the current process."""
        return int(self._mapping.get_env("global_rank", 0))

    @property
    def local_rank(self) -> int:
        """Return the rank of the current process."""
        return int(self._mapping.get_env("local_rank", self.global_rank))

    @property
    def master_addr(self) -> str:
        """Return the master address."""
        return self._mapping.get_env("master_addr", "")

    @property
    def master_port(self) -> int:
        """Return the master port."""
        return int(self._mapping.get_env("master_port", 0))

    @classmethod
    def used(cls) -> bool:
        return False
