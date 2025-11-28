# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging

from anemoi.inference.decorators import main_argument
from anemoi.inference.lazy import torch
from anemoi.inference.modifiers import modifier_registry
from anemoi.inference.modifiers.modifier import Modifier

LOG = logging.getLogger(__name__)


@modifier_registry.register("noise")
@main_argument("stddev")
class NoiseModifier(Modifier):
    """Model modifier that overrides the NoiseInjector std."""

    def __init__(self, context, stddev: int | float):
        super().__init__(context)
        self.stddev = stddev

    def modify(self, model: "torch.nn.Module") -> "torch.nn.Module":
        """Modify the given model by setting the stddev of NoiseInjector layers.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be modified.

        Returns
        -------
        torch.nn.Module
            The modified model with updated stddev for NoiseInjector layers.
        """
        try:
            from anemoi.models.layers.ensemble import NoiseInjector
        except ImportError as e:
            raise ImportError(
                "NoiseModifier requires anemoi models to be installed. " "Please install anemoi-models."
            ) from e

        count = 0

        for layer in model.modules():
            if isinstance(layer, NoiseInjector):
                layer.noise_std = self.stddev  # type: ignore
                count += 1

        LOG.info("Set noise stddev to %s for %d NoiseInjector layers.", self.stddev, count)
        return model
