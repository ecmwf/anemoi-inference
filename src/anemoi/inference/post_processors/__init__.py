# (C) Copyright 2025 ECMWF.
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

post_processor_registry = Registry(__name__)


def create_post_processor(context: Any, config: Dict[str, Any]) -> Any:
    return post_processor_registry.from_config(config, context)
