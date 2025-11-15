# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from anemoi.inference.clusters import cluster_registry
from anemoi.inference.clusters.mapping import EnvMapping
from anemoi.inference.clusters.mapping import MappingCluster

DISTRIBUTED_MAPPING = EnvMapping(
    local_rank="LOCAL_RANK",
    global_rank="RANK",
    world_size="WORLD_SIZE",
    master_addr="MASTER_ADDR",
    master_port="MASTER_PORT",
    init_method="env://",
)


@cluster_registry.register("distributed")
class DistributedCluster(MappingCluster):
    """Distributed cluster that uses environment variables for distributed setup."""

    def __init__(self) -> None:
        super().__init__(mapping=DISTRIBUTED_MAPPING)

    @classmethod
    def used(cls) -> bool:
        return bool(DISTRIBUTED_MAPPING.get_env("world_size")) and bool(DISTRIBUTED_MAPPING.get_env("global_rank"))
