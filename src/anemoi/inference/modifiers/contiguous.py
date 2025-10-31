# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from anemoi.inference.lazy import torch
from anemoi.inference.modifiers import modifier_registry
from anemoi.inference.modifiers.modifier import Modifier


@modifier_registry.register("contiguous")
class ContiguousModifier(Modifier):
    """Model modifier that ensures all weights are stored contiguously in memory."""

    def modify(self, model: "torch.nn.Module") -> "torch.nn.Module":
        """Modify the given model to ensure all weights are contiguous.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be modified.

        Returns
        -------
        torch.nn.Module
            The modified model with contiguous weights.
        """
        for param in model.parameters():
            param.data = param.data.contiguous()
        return model
