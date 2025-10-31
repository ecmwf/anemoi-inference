# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from abc import ABC
from abc import abstractmethod

from anemoi.inference.context import Context
from anemoi.inference.lazy import torch


class Modifier(ABC):
    """Abstract base class for model modifiers."""

    def __init__(self, context: Context):
        self.context = context

    @abstractmethod
    def modify(self, model: "torch.nn.Module") -> "torch.nn.Module":
        """Modify the given model.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be modified.

        Returns
        -------
        torch.nn.Module
            The modified model.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
