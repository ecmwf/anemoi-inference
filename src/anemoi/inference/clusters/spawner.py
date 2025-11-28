# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Callable

if TYPE_CHECKING:
    from anemoi.inference.clusters.client import ComputeClientFactory
    from anemoi.inference.config import Configuration

SPAWN_FUNCTION = Callable[["Configuration", "ComputeClientFactory"], None]


class ComputeSpawner(ABC):
    """Abstract base class for cluster operations for parallel execution."""

    @classmethod
    @abstractmethod
    def used(cls) -> bool:
        """Check if this client is valid in the current environment."""
        raise NotImplementedError

    @abstractmethod
    def spawn(self, fn: SPAWN_FUNCTION, config: "Configuration") -> None:
        """Spawn processes for parallel execution.

        Parameters
        ----------
        fn : SPAWN_FUNCTION
            The function to run in each process.
            Expects to receive the configuration and compute client factory as arguments.
        config : Configuration
            The configuration object for the runner.
        """
        raise NotImplementedError

    @abstractmethod
    def teardown(self) -> None:
        """Tear down the cluster environment."""
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.teardown()
