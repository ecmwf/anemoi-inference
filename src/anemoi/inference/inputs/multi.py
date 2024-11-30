# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from collections import defaultdict

import numpy as np

from ..input import Input
from . import create_input
from . import input_registry

LOG = logging.getLogger(__name__)


@input_registry.register("multi")
class MultiInput(Input):
    """An input to manage multiple sources of forcings."""

    def __init__(self, context, *args, inputs=None, **kwargs):
        super().__init__(context)
        if inputs is None:
            inputs = args
        assert isinstance(inputs, (list, tuple)), inputs
        self.inputs = {}
        self._input_per_id = {}
        for i in inputs:
            input = create_input(context, i["input"])
            self._input_per_id[id(input)] = input
            for v in i["variables"]:
                if v in self.inputs:
                    raise ValueError(f"Variable {v} already defined")
                self.inputs[v] = input

    def __repr__(self):
        return f"MultiInput({self.context.checkpoint.grid})"

    def create_input_state(self, *, date):
        raise NotImplementedError("MultiInput.create_input_state() not implemented")

    def load_forcings(self, variables, dates):
        inputs = defaultdict(list)
        for v in variables:
            if v not in self.inputs:
                raise ValueError(f"Variable {v} not defined")
            inputs[id(self.inputs[v])].append(v)

        rows = {}

        for input, vs in inputs.items():
            array = self._input_per_id[input].load_forcings(variables=vs, dates=dates)
            for i, v in enumerate(vs):
                rows[v] = array[i]

        return np.stack([rows[v] for v in variables], axis=0)
