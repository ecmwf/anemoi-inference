# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import base64
import logging

import earthkit.data as ekd

from ..input import Input
from . import input_registry

LOG = logging.getLogger(__name__)

TEMPLATE = """
R1JJQv//AAIAAAAAAAAA3AAAABUBAGIAABsBAQfoCRYGAAAAAQAAABECAAEAAQAJBAIwMDAxAAAA
SAMAAA/XoAAAAAAG////////////////////AAAFoAAAAtEAAAAA/////wVdSoAAAAAAMIVdSoAV
cVlwAAPQkAAD0JAAAAAAOgQAAAAIAcEC//8AAAABAAAAAAH//////////////wfoCRYGAAABAAAA
AAECAQAAAAD/AAAAAAAAABUFAA/XoAAAAAAAAIAKAAAAAAAAAAYG/wAAAAUHNzc3N0dSSUL//wAC
AAAAAAAAANwAAAAVAQBiAAAbAQEH6AkWDAAAAAEAAAARAgABAAEACQQBMDAwMQAAAEgDAAAP16AA
AAAABv///////////////////wAABaAAAALRAAAAAP////8FXUqAAAAAADCFXUqAFXFZcAAD0JAA
A9CQAAAAADoEAAAACAHBAv//AAAAAQAAAAAB//////////////8H6AkWDAAAAQAAAAABAgEAAAAA
/wAAAAAAAAAVBQAP16AAAAAAAACACgAAAAAAAAAGBv8AAAAFBzc3Nzc=
"""


@input_registry.register("templates")
class TemplatesInput(Input):
    """A dummy input that creates GRIB templates"""

    def __init__(self, context, *args, **kwargs):
        super().__init__(context)

    def create_input_state(self, *, date):
        raise NotImplementedError("TemplatesInput.create_input_state() not implemented")

    def template(self, variable, date, edition, **kwargs):

        # import eccodes
        typed = self.context.checkpoint.typed_variables[variable]

        if not typed.is_accumulation:
            return None

        template = base64.b64decode(TEMPLATE)
        # eccodes.codes_new_from_message(template)

        return ekd.from_source("memory", template)[0]

    def load_forcings(self, variables, dates):
        raise NotImplementedError("TemplatesInput.load_forcings() not implemented")
