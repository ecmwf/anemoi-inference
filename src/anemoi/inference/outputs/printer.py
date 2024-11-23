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

import numpy as np

from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)


def print_state(state, print=print):
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

    idx = list(range(0, len(names), len(names) // n))
    idx.append(len(names) - 1)
    idx = sorted(set(idx))

    length = max(len(name) for name in names)

    for i in idx:
        name = names[i]
        field = fields[name]
        min_value = f"min={np.nanmin(field):g}"
        max_value = f"max={np.nanmax(field):g}"
        print(f"    {name:{length}} shape={field.shape} {min_value:18s} {max_value:18s}")

    print()


@output_registry.register("printer")
class PrinterOutput(Output):
    """_summary_"""

    def write_initial_state(self, state):
        self.write_state(state)

    def write_state(self, state):
        print_state(state)
