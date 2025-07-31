# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

from ..context import Context
from ..input import Input
from ..runner import Runner
from . import runner_registry

LOG = logging.getLogger(__name__)


class CutoutInput(Input):
    """An Input object that combines two inputs."""

    def __init__(self, context: Context, lam: Input, globe: Input) -> None:
        """Initialize CutoutInput.

        Parameters
        ----------
        context : Context
            The context for the input.
        lam : Input
            The LAM input.
        globe : Input
            The globe input.
        """
        super().__init__(context)
        self.lam = lam
        self.globe = globe

    def create_input_state(self, *, date: str | None = None) -> None:
        """Create the input state.

        Parameters
        ----------
        date : Optional[str], optional
            The date for the input state, by default None
        """
        state1 = self.lam.create_input_state(date=date)
        state2 = self.globe.create_input_state(date=date)

        assert False, (state1, state2)


@runner_registry.register("cutout")
class CutoutRunner(Runner):
    """A Runner that for LAMs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize CutoutRunner."""
        super().__init__(*args, **kwargs)

        sources = self.checkpoint.sources

        if len(sources) != 2:
            raise ValueError(f"CutoutRunner expects two source, found {len(sources)}.")

        self.lam, self.globe = sources
