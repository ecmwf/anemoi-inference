# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from ..forcings import ComputedForcings
from ..forcings import Forcings
from ..runner import Runner
from . import runner_registry

LOG = logging.getLogger(__name__)

# This is because forcings are assumed to already be in the
# state dictionary, so we don't need to load them from anywhere.


class NoForcings(Forcings):
    """No forcings."""

    def __init__(self, context, variables, mask):
        super().__init__(context)
        self.variables = variables
        self.mask = mask
        self.kinds = dict(unknown=True)

    def load_forcings(self, state, date):
        pass


@runner_registry.register("simple")
class SimpleRunner(Runner):
    """Use that runner when using the low level API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_constant_computed_forcings(self, variables, mask):
        result = ComputedForcings(self, variables, mask)
        LOG.info("Constant computed forcing: %s", result)
        return result

    def create_dynamic_computed_forcings(self, variables, mask):
        result = ComputedForcings(self, variables, mask)
        LOG.info("Dynamic computed forcing: %s", result)
        return result

    def create_constant_coupled_forcings(self, variables, mask):
        # This runner does not support coupled forcings
        # there are supposed to be already in the state dictionary
        # of managed by the user.
        LOG.warning("Coupled forcings are not supported by this runner: %s", variables)
        return None

    def create_dynamic_coupled_forcings(self, variables, mask):
        # This runner does not support coupled forcings
        # there are supposed to be already in the state dictionary
        # of managed by the user.
        LOG.warning("Coupled forcings are not supported by this runner: %s", variables)
        return None
