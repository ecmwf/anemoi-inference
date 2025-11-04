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
from typing import Any


class ComputeSpawner(ABC):
    """Abstract base class for cluster operation."""

    @classmethod
    @abstractmethod
    def used(cls) -> bool:
        """Check if this client is valid in the current environment."""
        raise NotImplementedError

    @abstractmethod
    def spawn(self, fn: Any, *args: Any) -> None:
        """Spawn processes in the client environment.

        Should be overridden by subclasses if specific spawning logic is required.
        By default, this method does nothing.

        Parameters
        ----------
        fn : Any
            The function to run in each process.
        args : tuple[Any, ...]
            The arguments to pass to the function.
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
