# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np

from anemoi.inference.config import Configuration
from anemoi.inference.context import Context
from anemoi.inference.types import ProcessorConfig

from . import output_registry
from .masked import MaskedOutput

LOG = logging.getLogger(__name__)


@output_registry.register("extract_lam")
class ExtractLamOutput(MaskedOutput):
    """Extract LAM output class."""

    def __init__(
        self,
        context: Context,
        *,
        output: Configuration,
        lam: str = "lam_0",
        variables: list[str] | None = None,
        post_processors: list[ProcessorConfig] | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
    ) -> None:
        """Parameters
        ----------
        context : dict
            The context dictionary.
        output : dict
            The output configuration dictionary.
        lam : str, optional
            The LAM identifier, by default "lam_0".
        variables : list, optional
            The list of variables to extract, by default None.
        post_processors : Optional[List[ProcessorConfig]], default None
            Post-processors to apply to the input
        output_frequency : int, optional
            The frequency of output, by default None.
        write_initial_state : bool, optional
            Whether to write the initial state, by default None.
        """

        if "cutout_mask" in context.checkpoint.supporting_arrays:
            # Backwards compatibility
            mask = context.checkpoint.load_supporting_array("cutout_mask")
            points = slice(None, -np.sum(mask))
        else:
            if "lam_0" not in lam:
                raise NotImplementedError("Only lam_0 is supported")

            if "lam_1/cutout_mask" in context.checkpoint.supporting_arrays:
                raise NotImplementedError("Only lam_0 is supported")

            mask = context.checkpoint.load_supporting_array(f"{lam}/cutout_mask")

            assert len(mask) == np.sum(mask)
            points = slice(None, np.sum(mask))

        super().__init__(
            context,
            mask=points,
            output=output,
            variables=variables,
            post_processors=post_processors,
            output_frequency=output_frequency,
            write_initial_state=write_initial_state,
        )
