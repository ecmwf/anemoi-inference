# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging

from anemoi.inference.types import IntArray

from ..forcings import ComputedForcings
from ..forcings import Forcings
from ..runner import Runner
from . import runner_registry

LOG = logging.getLogger(__name__)


@runner_registry.register("plugin")
class PluginRunner(Runner):
    """A runner implementing the ai-models plugin API."""

    def __init__(self, checkpoint: str, *, device: str):
        """Initialize the PluginRunner.

        Parameters
        ----------
        checkpoint : str
            The checkpoint for the runner.
        device : str
            The device to run the model on.
        """
        super().__init__(checkpoint, device=device)

    # Compatibility with the ai_models API

    @property
    def param_sfc(self) -> list[str]:
        """Get surface parameters."""
        params, _ = self.checkpoint.mars_by_levtype("sfc")
        return sorted(params)

    @property
    def param_level_pl(self) -> tuple[list[str], list[int]]:
        """Get pressure level parameters and levels."""
        params, levels = self.checkpoint.mars_by_levtype("pl")
        return sorted(params), sorted(levels)

    @property
    def param_level_ml(self) -> tuple[list[str], list[int]]:
        """Get model level parameters and levels."""
        params, levels = self.checkpoint.mars_by_levtype("ml")
        return sorted(params), sorted(levels)

    @property
    def lagged(self) -> list[datetime.timedelta]:
        """Get lagged times in hours."""
        return self.checkpoint.lagged

    def create_constant_computed_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:
        """Create constant computed forcings.

        Parameters
        ----------
        variables : List[str]
            The variables for the computed forcings.
        mask : IntArray
            The mask for the computed forcings.

        Returns
        -------
        List[Forcings]
            The constant computed forcings.
        """
        result = ComputedForcings(self, variables, mask)
        LOG.info("Constant computed forcing: %s", result)
        return [result]

    def create_dynamic_computed_forcings(self, variables: list[str], mask: IntArray) -> list[Forcings]:
        """Create dynamic computed forcings.

        Parameters
        ----------
        variables : list
            The variables for the computed forcings.
        mask : IntArray
            The mask for the computed forcings.

        Returns
        -------
        List[Forcings]
            The dynamic computed forcings.
        """
        result = ComputedForcings(self, variables, mask)
        LOG.info("Dynamic computed forcing: %s", result)
        return [result]
