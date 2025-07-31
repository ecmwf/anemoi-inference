# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from anemoi.inference.config import Configuration
from anemoi.inference.context import Context
from anemoi.inference.types import ProcessorConfig

from . import output_registry
from .masked import MaskedOutput

LOG = logging.getLogger(__name__)


@output_registry.register("apply_mask")
class ApplyMaskOutput(MaskedOutput):
    """Apply mask output class."""

    def __init__(
        self,
        context: Context,
        *,
        mask: str,
        output: Configuration,
        variables: list[str] | None = None,
        post_processors: list[ProcessorConfig] | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
    ) -> None:
        """Parameters
        ----------
        context : dict
            The context dictionary.
        mask : str
            The mask identifier.
        output : dict
            The output configuration dictionary.
        post_processors : Optional[List[ProcessorConfig]], default None
            Post-processors to apply to the input
        output_frequency : int, optional
            The frequency of output, by default None.
        write_initial_state : bool, optional
            Whether to write the initial state, by default None.
        """
        super().__init__(
            context,
            mask=context.checkpoint.load_supporting_array(mask),
            output=output,
            variables=variables,
            post_processors=post_processors,
            output_frequency=output_frequency,
            write_initial_state=write_initial_state,
        )
