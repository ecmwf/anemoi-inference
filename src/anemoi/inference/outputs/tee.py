# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)


@output_registry.register("tee")
class TeeOutput(Output):
    """_summary_"""

    def __init__(self, context, outputs, *args, **kwargs):
        super().__init__(context)
        assert isinstance(outputs, list)
        self.outputs = []
        for output in outputs:
            LOG.info(f"Creating output {output}")
            if isinstance(output, str):
                output = {"kind": output}
            self.outputs.append(output_registry.create(output.pop("kind"), context, **output))

    def write_initial_state(self, state):
        for output in self.outputs:
            output.write_initial_state(state)

    def write_state(self, state):
        for output in self.outputs:
            output.write_state(state)
