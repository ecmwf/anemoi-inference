# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from functools import partial
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Union

import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.types import State

from ..decorators import ensure_path
from ..decorators import main_argument
from ..decorators import supports_parallel_output
from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)

ListOrAll = Union[list[str], Literal["all"]]


@output_registry.register("printer")
@main_argument("max_lines")
@ensure_path("path")
@supports_parallel_output("path")
class PrinterOutput(Output):
    """Printer output class."""

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        path: Path | None = None,
        variables: ListOrAll | None = None,
        max_lines: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initialise the PrinterOutput.

        Parameters
        ----------
        context : Context
            The context.
        metadata : Metadata
            Metadata corresponding to the dataset this output is handling.
        path : Path, optional
            The path to save the printed output, by default None.
            If the parent directory does not exist, it will be created.
        variables : list, optional
            The list of variables to print, by default None (max_lines will be printed). Can be also be the string "all", in which case all variables will be printed (regardless of max_lines).
        max_lines : int, optional
            The maximum number of lines to print, by default 4.
            If set to 0, all variables will be printed. If any value is provided in `variables`, this argument is ignored -- it is only used if `variables == None`.
        **kwargs : Any
            Additional keyword arguments.
        """
        # If "all" variables are included, self.variables should be None and max_lines should be 0, meaning
        # all variables are printed out.
        all_variables = variables == "all"

        super().__init__(context, metadata, variables=(None if all_variables else variables), **kwargs)

        self.max_lines = 0 if all_variables else max_lines

        self.print = print
        self.f = None

        if path is not None:
            self.f = open(path, "w")
            self.print = partial(print, file=self.f)

    def write_step(self, state: State) -> None:
        """Write a step of the state.

        Parameters
        ----------
        state : State
            The state dictionary.
        """
        self.print()
        if self.metadata.multi_dataset:
            self.print(f"[{self.dataset_name}]", end=" ")
        self.print_state(state)

    def print_state(self, state: State) -> None:
        """Print the state.

        Parameters
        ----------
        state : State
            The state dictionary.
        """
        print("😀", end=" ")
        for key, value in state.items():
            if isinstance(value, datetime.datetime):
                print(f"{key}={value.isoformat()}", end=" ")

            if isinstance(value, (str, float, int, bool, type(None))):
                print(f"{key}={value}", end=" ")

            if isinstance(value, np.ndarray):
                print(f"{key}={value.shape}", end=" ")

        fields = state.get("fields", {})

        print(f"fields={len(fields)}")
        print()

        names = list(fields.keys())
        selected = names

        if self.max_lines > 0:
            if self.variables is None:
                selected = names[: self.max_lines]
            else:
                LOG.debug(
                    f"Printer output settings contain a list of selected variables and a max_lines of {self.max_lines}. Ignoring the max_lines setting."
                )

        length = max((len(name) for name in names), default=0)

        for name in selected:
            if self.skip_variable(name):
                continue
            field = fields[name]
            min_value = f"min={np.nanmin(field):g}"
            max_value = f"max={np.nanmax(field):g}"
            print(f"    {name:{length}} shape={field.shape} {min_value:18s} {max_value:18s}")

        print()

    def close(self) -> None:
        if self.f is not None:
            self.f.close()
        return super().close()
