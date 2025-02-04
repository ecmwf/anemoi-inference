# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import earthkit.data as ekd

from . import TemplateProvider
from . import template_provider_registry

LOG = logging.getLogger(__name__)


@template_provider_registry.register("file")
class FileTemplates(TemplateProvider):
    """Template provider using a single GRIB file."""

    def __init__(self, manager, path):
        self.manager = manager
        self.path = path

    def template(self, grib, lookup):
        return ekd.from_source("file", self.path)[0]
