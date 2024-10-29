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
import numpy as np

from .grib import GribOutput

LOG = logging.getLogger(__name__)


class GribFileOutput(GribOutput):
    """
    Handles grib files
    """

    def __init__(self, path, checkpoint, *, allow_nans=False, **kwargs):
        super().__init__(checkpoint, allow_nans=allow_nans)
        self.path = path
        self.output = ekd.new_grib_output(self.path, split_output=True, **kwargs)

    def write_message(self, message, *args, **kwargs):
        try:
            self.output.write(message, *args, check_nans=self.allow_nans, **kwargs)
        except Exception as e:
            LOG.error("Error writing message to %s: %s", self.path, e)
            if np.isnan(message.data).any():
                LOG.error("Message contains NaNs (%s)", kwargs)
            raise e
