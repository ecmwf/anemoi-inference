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
from functools import partial

import numpy as np

from ..decorators import main_argument
from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)


def print_state(state, print=print, max_lines=4, variables=None):
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
    n = 4

    idx = list(range(0, len(names), max(1, len(names) // n)))
    idx.append(len(names) - 1)
    idx = sorted(set(idx))
    if variables == "all":
        variables = names
        max_lines = 0

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
    """_summary_"""

    def __init__(self, context, path=None, variables=None, **kwargs):
        super().__init__(context)
        self.print = print
        self.variables = variables
        assert variables == "all", variables
        if path is not None:
            self.f = open(path, "w")
            self.print = partial(print, file=self.f)

    def write_initial_state(self, state):
        self.write_state(state)

    def write_state(self, state):
        print_state(state, print=self.print, variables=self.variables)
