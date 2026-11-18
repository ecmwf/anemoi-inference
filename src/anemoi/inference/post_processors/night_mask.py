# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from earthkit.data import FieldList

from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata

from . import post_processor_registry
from .transform_filter import TransformFilter

LOG = logging.getLogger(__name__)


@post_processor_registry.register("night_mask")
class NightMask(TransformFilter):
    """Set accumulated fields to zero where the sun was down for the whole accumulation window.

    A field at time ``t`` accumulated over ``(t - accumulation_period, t]`` can only be non-zero if the sun was
    up at some point in that window. The fields are therefore set to zero where the mask variable (by default
    ``insolation``) is ``<= 0`` both at ``t`` and at ``t - accumulation_period``.

    This is the ``night_mask`` filter of anemoi-transform, with defaults taken from the checkpoint. It is
    independent of the runner: it works on any state whose fields are accumulated over a fixed period, for
    example the radiation fields of a forecast or of a temporal downscaler.

    It must come **before** any processor that accumulates the fields, such as
    ``accumulate_from_start_of_forecast``.

    Parameters
    ----------
    context : Context
        The context in which the processor is running.
    metadata : Metadata
        Metadata corresponding to the dataset this processor is handling.
    variables : list[str]
        The fields to set to zero at night.
    mask_variable : str, optional
        The computed forcing used as mask, by default "insolation".
    accumulation_period : str or int, optional
        The period over which the fields are accumulated, by default the data frequency of the checkpoint
        (the output timestep of a temporal downscaler).
    skip_initial_state : bool, optional
        Leave the initial state (step 0) unchanged, by default True: it holds no accumulation.
    """

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        variables: list[str],
        mask_variable: str = "insolation",
        accumulation_period: str | int | None = None,
        skip_initial_state: bool = True,
    ) -> None:
        if accumulation_period is None:
            accumulation_period = metadata.data_frequency

        super().__init__(
            context,
            metadata,
            filter={
                "night_mask": dict(
                    variables=list(variables),
                    mask_variable=mask_variable,
                    accumulation_period=accumulation_period,
                )
            },
            skip_initial_state=skip_initial_state,
        )
        LOG.info("Night mask on %s where %s <= 0 at t and t-%s", list(variables), mask_variable, accumulation_period)

    def _exec_filter(self, state: FieldList) -> FieldList:
        return self.filter.forward(state)

    def __repr__(self) -> str:
        """Return a string representation of the NightMask object."""
        return f"NightMask(filter={self.filter})"
