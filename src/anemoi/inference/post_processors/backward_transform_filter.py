# (C) Copyright 2024-2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any
from typing import Dict

from anemoi.transform.filters import filter_registry

from ..processor import Processor
from .earthkit_state import unwrap_state
from .earthkit_state import wrap_state

LOG = logging.getLogger(__name__)


class BackwardTransformFilter(Processor):

    def __init__(self, context: Any, filter: str, **kwargs: Any) -> None:
        super().__init__(context)
        self.filter: Any = filter_registry.create(filter, **kwargs)

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return unwrap_state(self.filter.backward(wrap_state(state)), state)
