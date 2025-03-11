# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import numpy as np

from ..input import Input
from . import create_input
from . import input_registry

LOG = logging.getLogger(__name__)


@input_registry.register("cutout")
class Cutout(Input):
    """Combines one or more LAMs into a global source using cutouts."""

    def __init__(self, context, **sources: dict[str, dict]):
        """Create a cutout input from a list of sources.

        Parameters
        ----------
        context : dict
            The context runner.
        sources : dict of sources
            A dictionary of sources to combine.
        """
        super().__init__(context)

        self.sources: dict[str, Input] = {}
        self.masks: dict[str, np.ndarray] = {}
        for src, cfg in sources.items():
            mask = cfg.pop("mask", f"{src}/cutout_mask")
            self.sources[src] = create_input(context, cfg)
            self.masks[src] = self.sources[src].checkpoint.load_supporting_array(mask)

    def __repr__(self):
        return f"Cutout({self.sources})"

    def create_input_state(self, *, date=None):
        """Create the input state dictionary."""

        LOG.info(f"Concatenating states from {self.sources}")
        sources = list(self.sources.keys())

        state = self.sources[sources[0]].create_input_state(date=date)
        for source in sources[1:]:
            mask = self.masks[source]
            _state = self.sources[source].create_input_state(date=date)

            state["latitudes"] = np.concatenate([state["latitudes"], _state["latitudes"][..., mask]], axis=-1)
            state["longitudes"] = np.concatenate([state["longitudes"], _state["longitudes"][..., mask]], axis=-1)
            for field, values in state["fields"].items():
                state["fields"][field] = np.concatenate([values, _state["fields"][field][..., mask]], axis=-1)

        return state

    def load_forcings(self, *, variables, dates):
        """Load forcings (constant and dynamic)."""
        forcings = []
        for source in self.sources:
            forcings.append(
                self.sources[source].load_forcings(variables=variables, dates=dates)[..., self.masks[source]]
            )
        forcings = np.concatenate(forcings, axis=-1)
        return forcings
