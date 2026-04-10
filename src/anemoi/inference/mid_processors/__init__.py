# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from anemoi.utils.registry import Registry

from anemoi.inference.context import Context
from anemoi.inference.processor import Processor
from anemoi.inference.types import ProcessorConfig

mid_processor_registry: Registry[Processor] = Registry(__name__)


def create_mid_processor(context: Context, config: ProcessorConfig) -> Processor:
    """Create a mid-processor, applied after each inference step.

    Parameters
    ----------
    context : Context
        The context for the mid-processor.
    config : Configuration
        The configuration for the mid-processor.

    Returns
    -------
    Processor
        The created mid-processor.
    """
    return mid_processor_registry.from_config(config, context)
