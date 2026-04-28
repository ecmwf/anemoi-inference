# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import tempfile

import earthkit.data as ekd
import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.types import State

from ..processor import Processor
from . import pre_processor_registry
from .forward_transform_filter import ForwardTransformFilter

LOG = logging.getLogger(__name__)


@pre_processor_registry.register("mask")
class MaskValues(Processor):
    """Replace values in a field with nans from a specified mask in the supporting arrays."""

    def __init__(self, context: Context, *, mask: str, param: str | list[str], **kwargs) -> None:
        """Initialize the MaskValues processor.

        Parameters
        ----------
        context : Context
            The context in which the processor operates.
        mask : str
            The mask to apply to the fields from the supporting arrays or a path to a mask file.
            Can be npy, grib, or netCDF file.
        param : str or list of str
            The parameter(s) to which the mask should be applied.
        kwargs : Any
            Additional keyword arguments to pass to the `apply_mask_to_param` filter.
            - `mask_value`: The value in the mask that indicates where to apply the mask (default: None).
            - `threshold_operator`: The operator to use for thresholding the mask if using a threshold (default: None).
            - `threshold`: The threshold value for the mask if using operator (default: None).
        """
        super().__init__(context)  # type: ignore
        if mask in self.checkpoint.supporting_arrays:  # type: ignore
            mask_array = self.checkpoint.supporting_arrays[mask]  # type: ignore
        elif mask.endswith(".npy"):
            mask_array = np.load(mask)
        elif mask.endswith(".grib") or mask.endswith(".nc"):
            mask_array = ekd.from_source("file", mask)[0].to_numpy(flatten=True)  # type: ignore
        else:
            raise ValueError(
                f"Mask '{mask}' not found in supporting arrays nor does it appear to be a file, available arrays: {list(self.checkpoint.supporting_arrays.keys())}."  # type: ignore
            )

        with tempfile.NamedTemporaryFile(suffix=".npy") as tmp_file:
            np.save(tmp_file.name, mask_array)
            self.filter = ForwardTransformFilter(
                context, filter="apply_mask", path=tmp_file.name, param=param, **kwargs
            )

    def process(self, state: State) -> State:
        """Apply the mask to the specified parameters in the state.

        Parameters
        ----------
        state : State
            The state containing the fields to be masked.

        Returns
        -------
        State
            The state with the specified parameters masked.
        """
        return self.filter.process(state)
