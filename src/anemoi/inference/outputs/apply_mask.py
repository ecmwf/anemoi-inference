# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Optional

from anemoi.inference.config import Configuration
from anemoi.inference.context import Context

from . import output_registry
from .masked import MaskedOutput

LOG = logging.getLogger(__name__)


@output_registry.register("apply_mask")
class ApplyMaskOutput(MaskedOutput):
    """Apply mask output class.

    Parameters
    ----------
    context : dict
        The context dictionary.
    mask : str
        The mask identifier.
    output : dict
        The output configuration dictionary.
    output_frequency : int, optional
        The frequency of output, by default None.
    write_initial_state : bool, optional
        Whether to write the initial state, by default None.
    """

    def __init__(
        self,
        context: Context,
        *,
        mask: str,
        output: Configuration,
        output_frequency: Optional[int] = None,
        write_initial_state: Optional[bool] = None,
    ) -> None:
        super().__init__(
            context,
            mask=context.checkpoint.load_supporting_array(mask),
            output=output,
            output_frequency=output_frequency,
            write_initial_state=write_initial_state,
        )
