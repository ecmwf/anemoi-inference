# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from anemoi.utils.registry import Registry
from icecream import ic

input_hres_registry = Registry(__name__)


def create_input_hres(context, config):
    ic(context, config)
    return input_hres_registry.from_config(config, context)
