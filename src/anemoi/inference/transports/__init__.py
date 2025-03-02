# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import Any

from anemoi.utils.registry import Registry

from anemoi.inference.config import Configuration

transport_registry = Registry(__name__)


def create_transport(config: Configuration, couplings: Any, tasks: Any) -> Any:
    """Create a transport instance based on the given configuration.

    Parameters
    ----------
    config : Config
        The configuration for the transport.
    couplings : Any
        The couplings for the transport.
    tasks : Any
        The tasks to be executed.

    Returns
    -------
    Any
        The created transport instance.
    """
    return transport_registry.from_config(config, couplings, tasks)
