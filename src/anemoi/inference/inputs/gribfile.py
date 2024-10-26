# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging

import earthkit.data as ekd

from .grib import GribInput

LOG = logging.getLogger(__name__)


class GribFileInput(GribInput):
    """
    Handles grib files
    """

    def __init__(self, path, checkpoint, *, use_grib_paramid=False):
        super().__init__(checkpoint, use_grib_paramid=use_grib_paramid)
        self.path = path

    def create_input_state(self, *, date):
        return self._create_input_state(ekd.from_source("file", self.path), date=date)
