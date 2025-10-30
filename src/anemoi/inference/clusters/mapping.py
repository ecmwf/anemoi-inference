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

from anemoi.inference.clusters import cluster_registry
from anemoi.inference.clusters.client import ComputeClient

LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class EnvMapping:
    local_rank: str
    global_rank: str
    world_size: str

    master_addr: str
    master_port: str

    backend: str | None = None
    init_method: str = "env://"


@cluster_registry.register("custom")  # type: ignore
class MappingCluster(ComputeClient):
    """Custom cluster that uses user-defined environment variables for distributed setup.

    Example usage
    -------------

    In the config
    ```yaml
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
    cluster = MappingCluster(context, mapping={
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
        return int(os.environ.get(self._mapping.world_size, 1))

    @property
    def global_rank(self) -> int:
        """Return the rank of the current process."""
        return int(os.environ.get(self._mapping.global_rank, 0))

    @property
    def local_rank(self) -> int:
        """Return the rank of the current process."""
        return int(os.environ.get(self._mapping.local_rank, self.global_rank))

    @property
    def master_addr(self) -> str:
        """Return the master address."""
        return os.environ.get(self._mapping.master_addr, "")

    @property
    def master_port(self) -> int:
        """Return the master port."""
        return int(os.environ.get(self._mapping.master_port, 0))

    @classmethod
    def used(cls) -> bool:
        return False
