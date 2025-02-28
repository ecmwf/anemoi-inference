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
import os
import zlib

from . import IndexTemplateProvider
from . import template_provider_registry

LOG = logging.getLogger(__name__)

# 1 - Get a GRIB with mars: retrieve,param=tp,levtype=sfc,type=fc,step=6,target=data.grib,grid=0.25/0.25
# 2 - grib_set -s edition=2,packingType=grid_ccsds data.grib data.grib2
# 3 - grib_set -d 0 data.grib2 out.grib
# 4 - python -c 'import base64, sys, zlib;print(base64.b64encode(zlib.compress(open(sys.argv[1], "rb").read())))' out.grib

# tp in grib2, 0.25/0.25 grid


@template_provider_registry.register("builtin")
class BuiltinTemplates(IndexTemplateProvider):
    """Builtin templates provider."""

    def __init__(self, manager, index_path=None):

        if index_path is None:
            index_path = os.path.join(os.path.dirname(__file__), "builtin.yaml")

        super().__init__(manager, index_path)

    def load_template(self, grib, lookup):
        import earthkit.data as ekd

        template = zlib.decompress(base64.b64decode(grib))
        return ekd.from_source("memory", template)[0]
