# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any

from anemoi.utils.registry import Registry

from anemoi.inference.config import Configuration

runner_registry = Registry(__name__)


def create_runner(config: Configuration, **kwargs: Any) -> Any:
    """Create a runner instance based on the given configuration.

    Parameters
    ----------
    config : Configuration
        The configuration for the runner.
    kwargs : dict
        Additional arguments for the runner.

    Returns
    -------
    Any
        The created runner instance.
    """
    return runner_registry.from_config(config.runner, config, **kwargs)
