# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
from typing import Optional

import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.types import State

from ..decorators import main_argument
from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)


@output_registry.register("raw")
@main_argument("path")
class RawOutput(Output):
    """Raw output class."""

    def __init__(
        self,
        context: Context,
        path: str,
        template: str = "{date}.npz",
        strftime: str = "%Y%m%d%H%M%S",
        output_frequency: Optional[int] = None,
        write_initial_state: Optional[bool] = None,
    ) -> None:
        """Initialize the RawOutput class.

        Parameters
        ----------
        context : dict
            The context.
        path : str
            The path to save the raw output.
        template : str, optional
            The template for filenames, by default "{date}.npz".
        strftime : str, optional
            The date format string, by default "%Y%m%d%H%M%S".
        output_frequency : int, optional
            The frequency of output, by default None.
        write_initial_state : bool, optional
            Whether to write the initial state, by default None.
        """
        super().__init__(context, output_frequency=output_frequency, write_initial_state=write_initial_state)
        self.path = path
        self.template = template
        self.strftime = strftime

    def __repr__(self) -> str:
        """Return a string representation of the RawOutput object.

        Returns
        -------
        str
            String representation of the RawOutput object.
        """
        return f"RawOutput({self.path})"

    def write_step(self, state: State) -> None:
        """Write the state to a compressed .npz file.

        Parameters
        ----------
        state : State
            The state to be written.
        """
        os.makedirs(self.path, exist_ok=True)
        date = state["date"].strftime(self.strftime)
        fn_state = f"{self.path}/{self.template.format(date=date)}"
        restate = {f"field_{key}": val for key, val in state["fields"].items()}
        for key in ["date"]:
            restate[key] = np.array(state[key], dtype=str)
        for key in ["latitudes", "longitudes"]:
            restate[key] = np.array(state[key])
        np.savez_compressed(fn_state, **restate)
