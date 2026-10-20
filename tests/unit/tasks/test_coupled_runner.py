# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.inference.config.couple import CoupleConfiguration
from anemoi.inference.tasks import create_task
from anemoi.inference.testing import fake_checkpoints
from anemoi.inference.testing import files_for_tests
from anemoi.inference.transports import create_transport


@pytest.mark.parametrize("runner", ["forecaster", "no-model"])
@fake_checkpoints
def test_runner_task_coupled_atmos_ocean(runner: str) -> None:
    config = CoupleConfiguration.load(files_for_tests("unit/configs/coupled.yaml"))
    global_config = {"runner": runner, "lead_time": "12h"}

    tasks = {name: create_task(name, action, global_config=global_config) for name, action in config.tasks.items()}
    transport = create_transport(config.transport, config.couplings, tasks)

    transport.start()
    transport.wait()
