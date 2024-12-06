# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from anemoi.utils.dates import frequency_to_timedelta as to_timedelta

from anemoi.inference.config import load_config
from anemoi.inference.runners.default import DefaultRunner

from ..task import Task
from . import task_registry

LOG = logging.getLogger(__name__)


@task_registry.register("input")
class InputTask(Task):
    """_summary_"""

    def __init__(self, name, config):
        super().__init__(name)
        self.config = load_config(config, [])

    def run(self, transport):
        LOG.info("Running task %s", self.name)
        couplings = transport.couplings(self)

        runner = DefaultRunner(self.config)

        inputs = runner.create_dynamic_coupled_forcings(
            ["lsm", "10u", "10v", "2d", "2t", "msl", "sf", "ssrd", "strd", "tcc", "tp"], 0
        )

        date = self.config.date
        last = self.config.date + to_timedelta(self.config.lead_time)
        dates = [date + h for h in runner.checkpoint.lagged]

        tag = 0
        while date <= last:
            LOG.info(f"=============== Loading: {dates}")
            for input in inputs:
                tensor = input.load_forcings_state({}, dates)
                LOG.info(f"Sending matrix: {tensor.shape} {tensor.size * tensor.itemsize}")
                for c in couplings:
                    c.apply(self, transport, tensor, tag=tag)

            tag += 1
            date += runner.checkpoint.timestep
            dates = [date]
