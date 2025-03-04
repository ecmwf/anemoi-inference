# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import List
from typing import Tuple

from anemoi.inference.types import IntArray

from ..forcings import ComputedForcings
from ..forcings import Forcings
from ..runner import Runner
from . import runner_registry

LOG = logging.getLogger(__name__)


@runner_registry.register("plugin")
class PluginRunner(Runner):
    """A runner implementing the ai-models plugin API."""

    def __init__(self, checkpoint: str, *, device: str, pre_processors=None, post_processors=None):
        """Initialize the PluginRunner.

        Parameters
        ----------
        checkpoint : str
            The checkpoint for the runner.
        device : str
            The device to run the model on.
        pre_processors : optional
            Pre-processors for the runner.
        post_processors : optional
            Post-processors for the runner.
        """
        super().__init__(checkpoint, device=device, pre_processors=pre_processors, post_processors=post_processors)

    # Compatibility with the ai_models API

    @property
    def param_sfc(self) -> List[str]:
        """Get surface parameters.

        Returns
        -------
        List[str]
            The surface parameters.
        """
        params, _ = self.checkpoint.mars_by_levtype("sfc")
        return sorted(params)

    @property
    def param_level_pl(self) -> Tuple[List[str], List[int]]:
        """Get pressure level parameters and levels.

        Returns
        -------
        Tuple[List[str], List[int]]
            The pressure level parameters and levels.
        """
        params, levels = self.checkpoint.mars_by_levtype("pl")
        return sorted(params), sorted(levels)

    @property
    def param_level_ml(self) -> Tuple[List[str], List[int]]:
        params, levels = self.checkpoint.mars_by_levtype("ml")
        return sorted(params), sorted(levels)

    @property
    def lagged(self) -> List[int]:
        return [s.total_seconds() // 3600 for s in self.checkpoint.lagged]

    def create_constant_computed_forcings(self, variables: List[str], mask: IntArray) -> List[Forcings]:
        result = ComputedForcings(self, variables, mask)
        LOG.info("Constant computed forcing: %s", result)
        return [result]

    def create_dynamic_computed_forcings(self, variables: list, mask: IntArray) -> List[Forcings]:
        result = ComputedForcings(self, variables, mask)
        LOG.info("Dynamic computed forcing: %s", result)
        return [result]
