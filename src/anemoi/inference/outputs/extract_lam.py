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

import numpy as np

from anemoi.inference.config import Configuration
from anemoi.inference.context import Context

from . import output_registry
from .masked import MaskedOutput

LOG = logging.getLogger(__name__)


@output_registry.register("extract_lam")
class ExtractLamOutput(MaskedOutput):
    """Extract LAM output class.

    Parameters
    ----------
    context : dict
        The context dictionary.
    output : dict
        The output configuration dictionary.
    lam : str, optional
        The LAM identifier, by default "lam_0".
    output_frequency : int, optional
        The frequency of output, by default None.
    write_initial_state : bool, optional
        Whether to write the initial state, by default None.
    """

    def __init__(
        self,
        context: Context,
        *,
        output: Configuration,
        lam: str = "lam_0",
        output_frequency: Optional[int] = None,
        write_initial_state: Optional[bool] = None,
    ) -> None:

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
            output_frequency=output_frequency,
            write_initial_state=write_initial_state,
        )
