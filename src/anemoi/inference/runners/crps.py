# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from . import runner_registry
from .default import DefaultRunner

LOG = logging.getLogger(__name__)


@runner_registry.register("crps")
class CrpsRunner(DefaultRunner):
    def predict_step(self, model, input_tensor_torch, fcstep, **kwargs):
        return model.predict_step(input_tensor_torch, fcstep=fcstep)
