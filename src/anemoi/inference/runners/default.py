# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from ..runner import Runner
from . import runner_registry

LOG = logging.getLogger(__name__)


@runner_registry.register("default")
class DefaultRunner(Runner):
    """Default runner for single source."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
