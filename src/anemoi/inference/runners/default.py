# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from ..runner import Runner

LOG = logging.getLogger(__name__)


class DefaultRunner(Runner):
    """Default runner for single source."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.checkpoint.sources) > 1:
            raise ValueError(f"Only one source is supported {self.checkpoint.sources}. Select another runner.")

    def mars_input(self, **kwargs):
        from ..inputs.mars import MarsInput

        return MarsInput(self, **kwargs)
