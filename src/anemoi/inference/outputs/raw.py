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
import os

import numpy as np

from . import Output

LOG = logging.getLogger(__name__)



class RawOutput(Output):
    """_summary_"""
    def __init__(self, path):
        self.path = path

    def write_initial_state(self, state):
        self.write_state(state)

    def write_state(self, state):
        os.makedirs(self.path, exist_ok=True)
        fn_state = f"{self.path}/{state['date'].strftime("%Y%m%d_%H")}"
        restate = {f"field_{key}" : val for key, val in state['fields'].items() }
        restate['longitudes'] = state['longitudes']
        restate['latitudes'] = state['latitudes']
        np.savez_compressed(fn_state,**restate)

        
