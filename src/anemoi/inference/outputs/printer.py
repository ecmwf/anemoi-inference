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
from collections.abc import Callable
from functools import partial
from typing import Any
from typing import Literal
from typing import Union

import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State

from ..decorators import main_argument
from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)

ListOrAll = Union[list[str], Literal["all"]]


def print_state(
    state: State,
    print: Callable[..., None] = print,
    max_lines: int = 4,
    variables: ListOrAll | None = None,
) -> None:
    """Print the state.

    Parameters
    ----------
    state : State
        The state dictionary.
    print : function, optional
        The print function to use, by default print.
    max_lines : int, optional
        The maximum number of lines to print, by default 4.
    variables : list, optional
        The list of variables to print, by default None.
    """
    print()
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

    if variables == "all":
        variables = names
        max_lines = 0

    if variables is None:
        variables = names

    if not isinstance(variables, (list, tuple, set)):
        variables = [variables]

    variables = set(variables)

    n = max_lines

    if max_lines == 0 or max_lines >= len(names):
        idx = list(range(len(names)))
    else:
        idx = list(range(0, len(names), len(names) // n))
        idx.append(len(names) - 1)
        idx = sorted(set(idx))

    length = max(len(name) for name in names)

    for i in idx:
        name = names[i]
        if name not in variables:
            continue
        field = fields[name]
        min_value = f"min={np.nanmin(field):g}"
        max_value = f"max={np.nanmax(field):g}"
        print(f"    {name:{length}} shape={field.shape} {min_value:18s} {max_value:18s}")

    print()


@output_registry.register("printer")
@main_argument("max_lines")
class PrinterOutput(Output):
    """Printer output class."""

    def __init__(
        self,
        context: Context,
        post_processors: list[ProcessorConfig] | None = None,
        path: str | None = None,
        variables: ListOrAll | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the PrinterOutput.

        Parameters
        ----------
        context : Context
            The context.
        post_processors : Optional[List[ProcessorConfig]] = None
            Post-processors to apply to the input
        path : str, optional
            The path to save the printed output, by default None.
        variables : list, optional
            The list of variables to print, by default None.
        **kwargs : Any
            Additional keyword arguments.
        """

        super().__init__(context, variables=variables, post_processors=post_processors)
        self.print = print
        self.variables = variables

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
        print_state(state, print=self.print, variables=self.variables)
