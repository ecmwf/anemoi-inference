# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os

from anemoi.inference.commands.run import _run
from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runners import create_runner
from anemoi.inference.testing import fake_checkpoints

# from dummy import dummy_checkpoints


HERE = os.path.dirname(__file__)


@fake_checkpoints
def test_inference():
    config = RunConfiguration.load(os.path.join(HERE, "configs/simple.yaml"), overrides=dict(device="cpu"))
    runner = create_runner(config)
    _run(runner, config)


if __name__ == "__main__":
    test_inference()
