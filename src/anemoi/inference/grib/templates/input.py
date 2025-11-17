# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any

import earthkit.data as ekd

from . import TemplateProvider
from . import template_provider_registry

LOG = logging.getLogger(__name__)


@template_provider_registry.register("input")
class InputTemplates(TemplateProvider):
    """Use input field (prognostics and non-computed forcings) as the output GRIB template."""

    def template(
        self,
        variable: str,
        lookup: dict[str, Any],
        *,
        input_templates: dict[str, ekd.Field],
        **kwargs,
    ) -> ekd.Field | None:
        return input_templates.get(variable)
