# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from anemoi.inference.checkpoint import Checkpoint
from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runner import Runner

from . import runner_registry

LOG = logging.getLogger(__name__)


@runner_registry.register("auto")
class AutoRunnerFactory(Runner):
    """Automatically select the correct Runner from the metadata `task` setting.

    This processes the configuration and selects the `Runner` subclass to return based on the `task`
    variable in the metadata. Defaults to `forecaster`.
    """

    def __new__(cls, config: RunConfiguration, **kwargs) -> Runner:
        checkpoint = Checkpoint(
            config.checkpoint,
            patch_metadata=config.patch_metadata,
        )
        runner_name = checkpoint.task
        if runner_name == "auto":
            raise ValueError("Task defined in configuration cannot be 'auto'.")

        if runner_registry.lookup(runner_name, return_none=True) is None:
            raise ValueError(
                f"Runner '{runner_name}' from the metadata `task` field is not registered. "
                f"Registered runners: {runner_registry.registered}"
            )

        return runner_registry.from_config(runner_name, config, **kwargs)
