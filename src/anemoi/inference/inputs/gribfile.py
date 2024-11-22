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

from ..decorators import main_argument
from . import input_registry
from .grib import GribInput

LOG = logging.getLogger(__name__)


@input_registry.register("grib")
@main_argument("path")
class GribFileInput(GribInput):
    """
    Handles grib files
    """

    def __init__(self, context, path, *, namer=None, **kwargs):
        super().__init__(context, namer=namer, **kwargs)
        self.path = path

    def create_input_state(self, *, date):
        return self._create_input_state(ekd.from_source("file", self.path), variables=None, date=date)

    def load_forcings(self, *, variables, dates):
        return self._load_forcings(ekd.from_source("file", self.path), variables=variables, dates=dates)

    def template(self, variable, date, **kwargs):
        fields = ekd.from_source("file", self.path)
        data = self._find_variable(fields, variable)
        if len(data) == 0:
            return None
        return data[0]
