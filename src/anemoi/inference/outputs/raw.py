# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path

import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.types import State
from anemoi.inference.utils.templating import render_template

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
        dir: Path,
        template: str = "{date}.npz",
        strftime: str = "%Y%m%d%H%M%S",
        variables: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Initialise the RawOutput class.

        Parameters
        ----------
        context : dict
            The context.
        dir : Path
            The directory to save the raw output.
            If the parent directory does not exist, it will be created.
        template : str, optional
            The template for filenames, by default "{date}.npz".
            Variables available are `date`, `basetime` `step`.
        strftime : str, optional
            The date format string, by default "%Y%m%d%H%M%S".
        """
        super().__init__(context, variables=variables, **kwargs)
        self.dir = Path(dir)
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
        date = state["date"]
        basetime = date - state["step"]

        self.dir.mkdir(parents=True, exist_ok=True)

        format_info = {
            "date": date.strftime(self.strftime),
            "step": state["step"],
            "basetime": basetime.strftime(self.strftime),
        }

        fn_state = f"{self.dir}/{render_template(self.template, format_info)}"
        restate = {f"field_{key}": val for key, val in state["fields"].items() if not self.skip_variable(key)}

        for key in ["date"]:
            restate[key] = np.array(state[key], dtype=str)

        for key in ["latitudes", "longitudes"]:
            restate[key] = np.array(state[key])

        np.savez_compressed(fn_state, **restate)
