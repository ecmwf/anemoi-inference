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
from anemoi.inference.metadata import Metadata
from anemoi.inference.processor import Processor
from anemoi.inference.types import ProcessorConfig

mid_processor_registry: Registry[Processor] = Registry(__name__)


def create_mid_processor(context: Context, config: ProcessorConfig, metadata: Metadata, **kwargs) -> Processor:
    """Create a mid-processor, applied after each inference step.

    Parameters
    ----------
    context : Context
        The context for the mid-processor.
    config : ProcessorConfig
        The configuration for the mid-processor.
    metadata : Metadata
        Metadata corresponding to the dataset this mid-processor is handling.
    **kwargs : Any
        Additional keyword arguments to pass to the mid-processor constructor.

    Returns
    -------
    Processor
        The created mid-processor.
    """
    return mid_processor_registry.from_config(config, context, metadata, **kwargs)
