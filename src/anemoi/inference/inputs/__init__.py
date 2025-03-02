# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import Any

from anemoi.utils.registry import Registry

from anemoi.inference.config import Configuration
from anemoi.inference.context import Context

input_registry = Registry(__name__)


def create_input(context: Context, config: Configuration) -> Any:
    return input_registry.from_config(config, context)
