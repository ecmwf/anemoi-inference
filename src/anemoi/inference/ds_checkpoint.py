# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
from collections import defaultdict
from functools import cached_property

from anemoi.utils.checkpoints import load_metadata
from earthkit.data.utils.dates import to_datetime

from .ds_metadata import Metadata_0, Metadata_1
from .checkpoint import Checkpoint
from icecream import ic
LOG = logging.getLogger(__name__)


class Checkpoint_0(Checkpoint):    
    def __init__(self, path):
        ic("Checkpoint_0")  
        super().__init__(path)


    @cached_property
    def _metadata(self):
        try:
            return Metadata_0(*load_metadata(self.path, supporting_arrays=True))
        except Exception as e:
            LOG.warning("Version for not support `supporting_arrays` (%s)", e)
            return Metadata_0(load_metadata(self.path))

    @property
    def number_of_input_grid_points(self):
        return self._metadata.number_of_input_grid_points            
        
class Checkpoint_1(Checkpoint):    
    def __init__(self, path):
        ic("Checkpoint_1")  
        super().__init__(path)

    @cached_property
    def _metadata(self):
        try:
            return Metadata_1(*load_metadata(self.path, supporting_arrays=True))
        except Exception as e:
            LOG.warning("Version for not support `supporting_arrays` (%s)", e)
            return Metadata_1(load_metadata(self.path))       

    @property
    def number_of_input_grid_points(self):
        return self._metadata.number_of_input_grid_points      