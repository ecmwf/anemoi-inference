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
from ..parallel import get_parallel_info
from ..outputs import create_output

LOG = logging.getLogger(__name__)


@runner_registry.register("parallel")
class ParallelRunner(DefaultRunner):

    def __init__(self, context):
        super().__init__(context)
        global_rank, local_rank, world_size = get_parallel_info()
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size

    def predict_step(self, model, input_tensor_torch, fcstep, **kwargs):
        model_comm_group = kwargs.get("model_comm_group", None)
        if model_comm_group is None:
            return model.predict_step(input_tensor_torch)
        else:
            try:
                return model.predict_step(input_tensor_torch, model_comm_group)
            except TypeError as err:
                LOG.error("Please upgrade to a newer version of anemoi-models to use parallel inference")
                raise err

    def create_output(self):
        if self.global_rank == 0:
            output = create_output(self, self.config.output)
            LOG.info("Output: %s", output)
            return output
        else:
            output = create_output(self, "none")
            return output
