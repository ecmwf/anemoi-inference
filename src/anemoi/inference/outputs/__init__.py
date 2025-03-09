# (C) Copyright 2024 ECMWF.
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
from anemoi.inference.output import Output

output_registry = Registry(__name__)


def create_output(context: Context, config: Configuration) -> Output:
    """Create an output.

    Parameters
    ----------
    context : Context
        The context for the output.
    config : Configuration
        The configuration for the output.

    Returns
    -------
    object
        The created output.
    """
    return output_registry.from_config(config, context)
