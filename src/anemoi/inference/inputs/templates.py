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

# 1 - Get a GRIB with mars
# 2 - grib_set -d 0 data.grib out.grib
# 4 - python -c 'import base64, sys;print(base64.b64encode(open(sys.argv[1], "rb").read()))' out.grib

# tp in grib2, 0.25/0.25 grid
TEMPLATE = """
R1JJQv//AAIAAAAAAAAA4AAAABUBAGIAACEBAQfoCxcGAAAAAQAAABECAAEAAQAJBAIwMDAxAAAASAMAAA/Xo
AAAAAAG////////////////////AAAFoAAAAtEAAAAA/////wVdSoAAAAAAMBVxWXAKtsRwAAPQkAAD0JAAAA
AAOgQAAAAIAcEC/54AAAABAAAAAAH//////////////wfoCxcMAAABAAAAAAECAQAAAAYNAAABwgAAABkFAA/
XoAAqAAAAAIATAAAAAA4gAIAAAAAGBv8AAAAFBzc3Nzc=
"""


@input_registry.register("templates")
class TemplatesInput(Input):
    """A dummy input that creates GRIB templates"""

    def __init__(self, context, *args, **kwargs):
        super().__init__(context)

    def create_input_state(self, *, date):
        raise NotImplementedError("TemplatesInput.create_input_state() not implemented")

    def template(self, variable, date, **kwargs):

        # import eccodes
        typed = self.context.checkpoint.typed_variables[variable]

        if not typed.is_accumulation:
            return None

        template = base64.b64decode(TEMPLATE)
        # eccodes.codes_new_from_message(template)

        return ekd.from_source("memory", template)[0]

    def load_forcings(self, variables, dates):
        raise NotImplementedError("TemplatesInput.load_forcings() not implemented")
