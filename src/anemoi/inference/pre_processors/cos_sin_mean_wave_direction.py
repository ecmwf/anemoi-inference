# (C) Copyright 2024-2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from . import pre_processor_registry
from .forward_transform_filter import ForwardTransformFilter

LOG = logging.getLogger(__name__)


@pre_processor_registry.register("cos_sin_mean_wave_direction")
class CosSinMeanWaveDirection(ForwardTransformFilter):
    pass
