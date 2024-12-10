# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import warnings
from collections import defaultdict
from copy import deepcopy
LOG = logging.getLogger(__name__)



class DownscalingMixin:

    # Support for downscaling checkpoint

    def split(self):
        metadata_0 = deepcopy(self._metadata)
        metadata_1 = deepcopy(self._metadata)

        for key in ('data', 'model'):
                metadata_0['data_indices'][key]['input'] = metadata_0['data_indices'][key]['input_0']
                metadata_1['data_indices'][key]['input'] = metadata_1['data_indices'][key]['input_1']

        zip0 = metadata_0['dataset']['specific']['datasets'][0]
        metadata_0['dataset']['specific'] = zip0['datasets'][0]

        zip1 = metadata_1['dataset']['specific']['datasets'][0]
        metadata_1['dataset']['specific'] = zip1['datasets'][1]


        return [self.__class__(metadata_0), self.__class__(metadata_1)]
