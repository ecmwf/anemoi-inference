# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from ..context import Context
from ..input import Input
from ..runner import Runner
from . import runner_registry

LOG = logging.getLogger(__name__)


class CutoutInput(Input):
    """An Input object that combines two inputs."""

    def __init__(self, context, lam, globe):
        super().__init__(context)
        self.lam = lam
        self.globe = globe

    def create_input_state(self, *, date=None):

        state1 = self.lam.create_input_state(date=date)
        state2 = self.globe.create_input_state(date=date)

        assert False, (state1, state2)


class CutoutContext(Context):
    """A Context object for CutoutRunner."""

    def __init__(self, checkpoint):
        self._checkpoint = checkpoint

    @property
    def checkpoint(self):
        return self._checkpoint


@runner_registry.register("cutout")
class CutoutRunner(Runner):
    """A Runner that for LAMs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sources = self.checkpoint.sources

        if len(sources) != 2:
            raise ValueError(f"CutoutRunner expects two source, found {len(sources)}.")

        self.lam, self.globe = sources
