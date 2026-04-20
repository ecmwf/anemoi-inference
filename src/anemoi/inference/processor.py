# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

from anemoi.inference.types import DataRequest
from anemoi.inference.types import State

if TYPE_CHECKING:
    from anemoi.inference.context import Context


class Processor(ABC):
    """Abstract base class for processors.

    Parameters
    ----------
    context : Context
        The context in which the processor operates.
    """

    def __init__(self, context: "Context") -> None:
        self.context = context
        self.checkpoint = context.checkpoint

    def __repr__(self) -> str:
        """Return a string representation of the processor.

        Returns
        -------
        str
            The class name of the processor.
        """
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def process(self, state: State) -> State:
        """Process the given state.

        Parameters
        ----------
        state : State
            The state to be processed.

        Returns
        -------
        State
            The processed state.
        """
        pass

    def patch_data_request(self, data_request: DataRequest) -> DataRequest:
        """Override if a processor needs to patch the data request (e.g. mars or cds).

        Parameters
        ----------
        data_request : DataRequest
            The data request to be patched.

        Returns
        -------
        DataRequest
            The patched data request.
        """
        return data_request
