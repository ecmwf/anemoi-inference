# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import TYPE_CHECKING

LOG = logging.getLogger(__name__)


class LazyModule:
    """Defer loading of a module until attribute access."""

    def __init__(self, module_name: str):
        self.__module_name = module_name
        self.__module = None

    def __getattr__(self, name: str):
        if self.__module is None:
            import importlib

            LOG.debug("Importing '%s'", self.__module_name)

            self.__module = importlib.import_module(self.__module_name)
        return getattr(self.__module, name)


# add heavy imports here, then they can be used in the rest of the codebase as regular imports
# with type checking and autocompletion: `from anemoi.inference.lazy import torch`
# note: when used in a type hint, use quotes, e.g. "torch.Tensor" instead of torch.Tensor to avoid triggering the import
if TYPE_CHECKING:
    import torch
else:
    torch = LazyModule("torch")
