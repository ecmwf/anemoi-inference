# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import List
from typing import Optional

from anemoi.inference.input import Input
from anemoi.inference.output import Output
from anemoi.inference.processor import Processor
from anemoi.inference.types import IntArray

if TYPE_CHECKING:
    from .forcings import Forcings


LOG = logging.getLogger(__name__)


class Context(ABC):
    """Represents the context of the inference."""

    allow_nans = (None,)  # can be True of False
    use_grib_paramid = False
    verbosity = 0
    development_hacks = {}  # For testing purposes, don't use in production

    # Some runners will set these values, which can be queried by Output objects,
    # but may remain as None

    reference_date = None
    time_step = None
    lead_time = None
    output_frequency: Optional[int] = None
    write_initial_state: bool = True

    ##################################################################

    @property
    @abstractmethod
    def checkpoint(self):
        """Returns the checkpoint used for the inference."""
        pass

    ##################################################################
    # The methods below are not marked as abstract because they are
    # not required to be implemented by subclasses. However, they are
    # expected to be implemented by subclasses when relevant.
    # For example, when running the `SimpleRunner`, the user is
    # expected to provide the forcings directly as input to the runner.
    ##################################################################

    def create_input(self) -> Input:
        raise NotImplementedError()

    def create_output(self) -> Output:
        raise NotImplementedError()

    def create_constant_computed_forcings(self, variables: List[str], mask: IntArray) -> List["Forcings"]:
        raise NotImplementedError(f"{self.__class__.__name__}.create_constant_computed_forcings")

    def create_constant_coupled_forcings(self, variables: List[str], mask: IntArray) -> List["Forcings"]:
        raise NotImplementedError(f"{self.__class__.__name__}.create_constant_coupled_forcings")

    def create_dynamic_computed_forcings(self, variables: List[str], mask: IntArray) -> List["Forcings"]:
        raise NotImplementedError(f"{self.__class__.__name__}.create_dynamic_computed_forcings")

    def create_dynamic_coupled_forcings(self, variables: List[str], mask: IntArray) -> List["Forcings"]:
        raise NotImplementedError(f"{self.__class__.__name__}.create_dynamic_coupled_forcings")

    def create_pre_processors(self) -> List[Processor]:
        return []

    def create_post_processors(self) -> List[Processor]:
        return []
