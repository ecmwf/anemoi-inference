# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from anemoi.utils.registry import Registry

from anemoi.inference.config import Configuration
from anemoi.inference.context import Context

from .modifier import Modifier

modifier_registry = Registry(__name__)


def create_modifier(context: Context, config: Configuration) -> Modifier:
    """Create a modifier.

    Parameters
    ----------
    context : Context
        The context for the modifier.
    config : Configuration
        The configuration for the modifier.

    Returns
    -------
    object
        The created modifier.
    """
    return modifier_registry.from_config(config, context)
