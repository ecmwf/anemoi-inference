# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from anemoi.utils.registry import Registry

from anemoi.inference.context import Context

from .cluster import Cluster

cluster_registry: Registry[Cluster] = Registry(__name__)


def create_cluster(context: Context, config: dict, *args, **kwargs) -> Cluster:
    """Find and return the appropriate cluster for the current environment.

    Returns
    -------
    Cluster
        The found cluster instance.
    """
    if config:
        return cluster_registry.from_config(config, context, *args, **kwargs)

    for cluster in cluster_registry.factories:
        cluster_cls = cluster_registry.lookup(cluster)
        assert cluster_cls is not None

        if cluster_cls.used():
            return cluster_cls(context, *args, **kwargs)
    raise RuntimeError(
        f"No suitable cluster found for the current environment,\nDiscovered implementations were {cluster_registry.registered}."
    )
