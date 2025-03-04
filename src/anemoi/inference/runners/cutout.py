# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

from ..context import Context
from ..input import Input
from ..runner import Runner
from . import runner_registry

LOG = logging.getLogger(__name__)


class CutoutInput(Input):
    """An Input object that combines two inputs."""

    def __init__(self, context, lam, globe):
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

    def create_input_state(self, *, date: Optional[str] = None) -> None:
        """Create the input state.

        Parameters
        ----------
        date : Optional[str], optional
            The date for the input state, by default None
        """
        state1 = self.lam.create_input_state(date=date)
        state2 = self.globe.create_input_state(date=date)

        assert False, (state1, state2)


class CutoutContext(Context):
    """A Context object for CutoutRunner."""

    def __init__(self, checkpoint: str) -> None:
        """Initialize CutoutContext.

        Parameters
        ----------
        checkpoint : str
            The checkpoint for the context.
        """
        self._checkpoint = checkpoint

    @property
    def checkpoint(self) -> str:
        """Get the checkpoint.

        Returns
        -------
        str
            The checkpoint.
        """
        return self._checkpoint


@runner_registry.register("cutout")
class CutoutRunner(Runner):
    """A Runner that for LAMs."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize CutoutRunner."""
        super().__init__(*args, **kwargs)

        sources = self.checkpoint.sources

        if len(sources) != 2:
            raise ValueError(f"CutoutRunner expects two source, found {len(sources)}.")

        self.lam, self.globe = sources
