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
from anemoi.inference.clusters.mapping import EnvMapping
from anemoi.inference.clusters.mapping import MappingCluster
from anemoi.inference.context import Context

LOG = logging.getLogger(__name__)

DISTRIBUTED_MAPPING = EnvMapping(
    local_rank="LOCAL_RANK",
    global_rank="RANK",
    world_size="WORLD_SIZE",
    master_addr="MASTER_ADDR",
    master_port="MASTER_PORT",
    # backend="nccl",
    init_method="env://",
)


@cluster_registry.register("distributed")  # type: ignore
class DistributedCluster(MappingCluster):  # type: ignore
    """Distributed cluster that uses environment variables for distributed setup."""

    def __init__(self, context: Context, **kwargs) -> None:
        super().__init__(context, mapping=DISTRIBUTED_MAPPING, **kwargs)

    @classmethod
    def used(cls) -> bool:
        return DISTRIBUTED_MAPPING.global_rank in os.environ and DISTRIBUTED_MAPPING.local_rank in os.environ
