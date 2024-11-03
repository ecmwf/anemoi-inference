# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import earthkit.data as ekd

from . import input_registry
from .grib import GribInput

LOG = logging.getLogger(__name__)


@input_registry.register("grib")
class GribFileInput(GribInput):
    """
    Handles grib files
    """

    def __init__(self, context, path, *, use_grib_paramid=False):
        super().__init__(context, use_grib_paramid=use_grib_paramid)
        self.path = path

    def create_input_state(self, *, date):
        return self._create_input_state(ekd.from_source("file", self.path), variable=None, date=date)

    def load_forcings(self, *, variables, dates):
        return self._load_forcings(ekd.from_source("file", self.path), variables=variables, dates=dates)
