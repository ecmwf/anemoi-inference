# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import time

import numpy as np

from ..task import Task
from . import task_registry

LOG = logging.getLogger(__name__)


@task_registry.register("debug")
class DebugTask(Task):
    """_summary_"""

    def run(self, transport):
        LOG.info("Running task %s", self.name)
        couplings = transport.couplings(self.name)

        tensor = np.zeros(shape=(10, 10))
        for i in range(10):
            LOG.info("%r: Running iteration %s", self, i)

            tensor += 1

            for c in couplings:
                c.apply(self, transport, tensor, tag=i)

            time.sleep(1)

        LOG.info("Task %s finished", self.name)
