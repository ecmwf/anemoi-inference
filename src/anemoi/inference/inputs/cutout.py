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

    def __init__(self, context, **sources):
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
        for src, cfg in sources.items():
            self.sources[src] = create_input(context, cfg)

    def __repr__(self):
        return f"Cutout({self.sources})"

    def create_input_state(self, *, date=None):
        """Create the input state dictionary."""

        LOG.info(f"Concatenating states from {self.sources}")
        src = list(self.sources.keys())
        state = self.sources[src[0]].create_input_state(date=date)
        for i, source in enumerate(src[1:]):
            _state = self.sources[source].create_input_state(date=date)

            # NOTE: we must decide whether these come from the checkpoint or the input
            # state["latitudes"] = np.concatenate(
            #     [state["latitudes"], _state["latitudes"]], axis=-1
            # )
            # state["longitudes"] = np.concatenate(
            #     [state["longitudes"], _state["longitudes"]], axis=-1
            # )
            for field, values in state["fields"].items():
                state["fields"][field] = np.concatenate([values, _state["fields"][field]], axis=-1)

        return state

    def load_forcings(self, *, variables, dates):
        """Load forcings (constant and dynamic)."""
        forcings = []
        for source in self.sources.values():
            forcings.append(source.load_forcings(variables=variables, dates=dates))
        forcings = np.concatenate(forcings, axis=-1)
        return forcings
