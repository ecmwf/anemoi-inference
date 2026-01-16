# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
from typing import Any

from anemoi.utils.registry import Registry

from .client import ComputeClientFactory
from .spawner import ComputeSpawner

LOG = logging.getLogger(__name__)

cluster_registry: Registry[ComputeClientFactory | ComputeSpawner] = Registry(__name__)


def create_cluster(config: dict[str, Any] | str, *args, **kwargs) -> ComputeClientFactory | ComputeSpawner:
    """Find and return the appropriate cluster for the current environment.

    Parameters
    ----------
    config : dict
        Configuration for the cluster.
        Can be string or dict.
    args : Any
        Additional positional arguments.
    kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    Cluster
         The created cluster instance.
    """
    if config:
        return cluster_registry.from_config(config, *args, **kwargs)

    discovered_clusters = []

    for cluster in cluster_registry.factories:
        cluster_cls = cluster_registry.lookup(cluster)
        assert cluster_cls is not None

        if cluster_cls.used():
            discovered_clusters.append(cluster_cls)

    if len(discovered_clusters) == 1:
        cluster_cls = discovered_clusters[0]
        return cluster_cls(*args, **kwargs)

    elif len(discovered_clusters) > 1:
        LOG.warning(
            "Multiple suitable clusters found for the current environment: %s,\nWill sort, and select the one with the highest priority.",
            discovered_clusters,
        )
        discovered_clusters.sort(key=lambda cls: getattr(cls, "priority", 0), reverse=True)
        cluster_cls = discovered_clusters[0]
        LOG.warning("Selected cluster after sorting: %s", cluster_cls)
        return cluster_cls(*args, **kwargs)

    raise RuntimeError(
        f"No suitable cluster found for the current environment,\nDiscovered possible implementations were {cluster_registry.registered}."
    )
