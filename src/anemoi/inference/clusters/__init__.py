# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import Any

from anemoi.utils.registry import Registry

from .client import ComputeClientFactory
from .spawner import ComputeSpawner

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

    for cluster in cluster_registry.factories:
        cluster_cls = cluster_registry.lookup(cluster)
        assert cluster_cls is not None

        if cluster_cls.used():
            return cluster_cls(*args, **kwargs)

    raise RuntimeError(
        f"No suitable cluster found for the current environment,\nDiscovered implementations were {cluster_registry.registered}."
    )
