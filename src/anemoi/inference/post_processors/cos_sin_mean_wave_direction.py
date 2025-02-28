# (C) Copyright 2024-2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any
from typing import Dict

from anemoi.transform.filters import filter_registry

from ..processor import Processor
from . import post_processor_registry
from .state import unwrap_state
from .state import wrap_state

LOG = logging.getLogger(__name__)


@post_processor_registry.register("cos_sin_mean_wave_direction")
class CosSinMeanWaveDirection(Processor):
    """Processor for calculating the mean wave direction using cosine and sine components.

    Parameters
    ----------
    context : Any
        The context in which the processor is used.
    **kwargs : Any
        Additional keyword arguments for the filter.
    """

    def __init__(self, context: Any, **kwargs: Any):
        """Initialize the CosSinMeanWaveDirection processor.

        Parameters
        ----------
        context : Any
            The context in which the processor is used.
        **kwargs : Any
            Additional keyword arguments for the filter.
        """
        super().__init__(context)
        self.filter = filter_registry.create("cos_sin_mean_wave_direction", **kwargs)

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the state to calculate the mean wave direction.

        Parameters
        ----------
        state : Dict[str, Any]
            The state dictionary containing the data to be processed.

        Returns
        -------
        Dict[str, Any]
            The processed state with the mean wave direction.
        """
        return unwrap_state(self.filter.backward(wrap_state(state)), state)

    def patch_data_request(self, data_request: Any) -> Any:
        """Patch the data request to include necessary parameters for processing.

        Parameters
        ----------
        data_request : Any
            The data request object to be patched.

        Returns
        -------
        Any
            The patched data request.
        """
        return self.filter.patch_data_request(data_request)
