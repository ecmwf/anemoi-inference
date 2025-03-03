# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import Any
from typing import Dict

from anemoi.utils.registry import Registry

task_registry = Registry(__name__)


def create_task(name: str, config: Dict[str, Any], global_config: Dict[str, Any]) -> Any:
    """Create a task instance based on the given configuration.

    Parameters
    ----------
    name : str
        The name of the task.
    config : Dict[str, Any]
        The configuration for the task.
    global_config : Dict[str, Any]
        The global configuration.

    Returns
    -------
    Any
        The created task instance.
    """
    return task_registry.from_config(config, name, global_config=global_config)
