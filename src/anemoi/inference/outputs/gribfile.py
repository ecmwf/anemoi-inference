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

from .grib import GribOutput

LOG = logging.getLogger(__name__)


class GribFileOutput(GribOutput):
    """
    Handles grib files
    """

    def __init__(self, path, checkpoint, *, verbose=True, **kwargs):
        super().__init__(checkpoint, verbose=verbose)
        self.path = path
        self.output = ekd.new_grib_output(self.path, split_output=True, **kwargs)

    def write_message(self, message, *args, **kwargs):
        self.output.write(message, *args, **kwargs)
