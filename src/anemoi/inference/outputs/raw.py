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

import numpy as np

from ..decorators import main_argument
from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)


@output_registry.register("raw")
@main_argument("path")
class RawOutput(Output):
    """_summary_"""

    def __init__(
        self,
        context,
        path,
        template="{date}.npz",
        strftime="%Y%m%d%H%M%S",
    ):
        super().__init__(context)
        self.path = path
        self.template = template
        self.strftime = strftime

    def __repr__(self):
        return f"RawOutput({self.path})"

    def write_initial_state(self, state):
        reduced_state = self.reduce(state)
        self.write_state(reduced_state)

    def write_state(self, state):
        os.makedirs(self.path, exist_ok=True)
        date = state["date"].strftime(self.strftime)
        fn_state = f"{self.path}/{self.template.format(date=date)}"
        restate = {f"field_{key}": val for key, val in state["fields"].items()}
        for key in ["date"]:
            restate[key] = np.array(state[key], dtype=str)
        for key in ["latitudes", "longitudes"]:
            restate[key] = np.array(state[key])
        np.savez_compressed(fn_state, **restate)
