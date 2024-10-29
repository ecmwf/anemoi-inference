# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from ..context import Context
from ..inputs import Input
from ..runner import Runner

LOG = logging.getLogger(__name__)


class CutoutInput(Input):
    """An Input object that combines two inputs."""

    def __init__(self, context, lam, globe):
        self.lam = lam
        self.globe = globe

    def create_input_state(self, *, date=None):

        state2 = self.globe.create_input_state(date=date)
        state1 = self.lam.create_input_state(date=date)
        assert False, (state1, state2)


class CutoutContext(Context):
    """A Context object for CutoutRunner."""

    def __init__(self, checkpoint):
        self._checkpoint = checkpoint

    @property
    def checkpoint(self):
        return self._checkpoint


class CutoutRunner(Runner):
    """A Runner that for LAMs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sources = self.checkpoint.sources

        if len(sources) != 2:
            raise ValueError(f"CutoutRunner expects two source, found {len(sources)}.")

        self.lam, self.globe = sources

    def mars_input(self, **kwargs):
        from ..inputs.mars import MarsInput

        mars1 = MarsInput(CutoutContext(self.lam), **kwargs)
        mars2 = MarsInput(CutoutContext(self.globe), **kwargs)
        return CutoutInput(self, mars1, mars2)
