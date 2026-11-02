# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runners import create_runner
from anemoi.inference.testing import fake_checkpoints
from anemoi.inference.testing import files_for_tests


@pytest.mark.parametrize("config_name", ["simple", "mwd", "atmos", "ocean"])
@fake_checkpoints
def test_forecaster_runner_execute_fake_checkpoint(config_name: str) -> None:
    config = RunConfiguration.load(
        files_for_tests(f"unit/configs/{config_name}.yaml"),
        overrides=dict(runner="forecaster", device="cpu", input="dummy"),
    )
    runner = create_runner(config)
    runner.execute()
