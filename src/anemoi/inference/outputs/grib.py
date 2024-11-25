# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import json
import logging
from abc import abstractmethod

from earthkit.data.utils.dates import to_datetime

from ..grib.encoding import grib_keys
from ..inputs import create_input
from ..output import Output

LOG = logging.getLogger(__name__)


class GribOutput(Output):
    """
    Handles grib
    """

    def __init__(self, context, *, encoding=None, templates=None, grib1_keys=None, grib2_keys=None):
        super().__init__(context)
        self._first = True
        self.typed_variables = self.checkpoint.typed_variables
        self.quiet = set()
        self.encoding = encoding if encoding is not None else {}
        self.templates = templates
        self.grib1_keys = grib1_keys if grib1_keys is not None else {}
        self.grib2_keys = grib2_keys if grib2_keys is not None else {}
        self._template_cache = None
        self._template_source = None
        self._template_date = None
        self._template_reuse = None
        self.use_closest_template = False  # Off for now

    def write_initial_state(self, state):
        # We trust the GribInput class to provide the templates
        # matching the input state

        for name in state["fields"]:

            template = self.template(state, name)
            if template is None:
                # We can currently only write grib output if we have a grib input
                raise ValueError(
                    "GRIB output only works if the input is GRIB (for now). Set `write_initial_state` to `false`."
                )

            variable = self.typed_variables[name]
            if variable.is_accumulation:
                LOG.warning("Found accumulated variable `%s` is initial state.", name)

            # assert False,
            # values = state["fields"][name]
            # assert False, values.shape
            values = None
            keys = grib_keys(
                values=values,
                template=template,
                accumulation=variable.is_accumulation,
                param=None,
                date=None,
                time=None,
                step=0,
                keys=self.encoding,
                grib1_keys=self.grib1_keys,
                grib2_keys=self.grib2_keys,
                quiet=self.quiet,
            )

            # LOG.info("Step 0 GRIB %s\n%s", template, json.dumps(keys, indent=4))

            self.write_message(values, template=template, **keys)

    def write_state(self, state):

        reference_date = self.context.reference_date
        date = state["date"]

        step = date - reference_date
        step = step.total_seconds() / 3600
        assert int(step) == step, step
        step = int(step)

        if "_grib_templates_for_output" not in state:
            if "_grib_templates_for_output" not in self.quiet:
                self.quiet.add("_grib_templates_for_output")
                LOG.warning("Input is not GRIB.")

        for name, value in state["fields"].items():
            keys = {}

            variable = self.typed_variables[name]
            param = variable.grib_keys.get("param", variable)

            template = self.template(state, name)

            if template is None:
                if name not in self.quiet:
                    LOG.warning("No GRIB template found for `%s`. This may lead to unexpected results.", name)
                    self.quiet.add(name)

                variable_keys = variable.grib_keys.copy()
                for key in ("class", "type", "stream", "expver", "date", "time", "step"):
                    variable_keys.pop(key, None)

                keys.update(variable_keys)

            keys.update(self.encoding)

            keys = grib_keys(
                values=value,
                template=template,
                date=reference_date.strftime("%Y-%m-%d"),
                time=reference_date.hour,
                step=step,
                param=param,
                accumulation=variable.is_accumulation,
                keys=keys,
                grib1_keys=self.grib1_keys,
                grib2_keys=self.grib2_keys,
                quiet=self.quiet,
            )

            if LOG.isEnabledFor(logging.DEBUG):
                LOG.info("Encoding GRIB %s\n%s", template, json.dumps(keys, indent=4))

            try:
                self.write_message(value, template=template, **keys)
            except Exception:
                LOG.error("Error writing field %s", name)
                LOG.error("Template: %s", template)
                LOG.error("Keys:\n%s", json.dumps(keys, indent=4))
                raise

    @abstractmethod
    def write_message(self, message, *args, **kwargs):
        pass

    def template(self, state, name):

        if self._template_cache is None:
            self._template_cache = {}
            if "_grib_templates_for_output" in state:
                self._template_cache.update(state.get("_grib_templates_for_output", {}))

        if name is None:
            return self._template_cache

        if name in self._template_cache:
            return self._template_cache[name]

        # Catch all template
        if None in self._template_cache:
            return self._template_cache[None]

        if self.use_closest_template:  #
            template, name2 = self._clostest_template(self._template_cache, name)

            if name not in self.quiet:
                if name2 is not None:
                    LOG.warning("Using template for `%s`", name2)
                self.quiet.add(name)

                self._template_cache[name] = template
                return template

        if not self.templates:
            return None

        if self._template_source is None:
            if "source" not in self.templates:
                raise ValueError("No `source` given in `templates`")

            self._template_source = create_input(self.context, self.templates["source"])
            LOG.info("Loading templates from %s", self._template_source)

            if "date" in self.templates:
                self._template_date = to_datetime(self.templates["date"])

            self._template_reuse = self.templates.get("reuse", False)

        LOG.info("Loading template for %s from %s", name, self._template_source)

        date = self._template_date if self._template_date is not None else state["date"]
        field = self._template_source.template(variable=name, date=date)

        if field is None:
            LOG.warning("No template found for `%s`", name)

        if self._template_reuse:
            self._template_cache[None] = field
        else:
            self._template_cache[name] = field

        return field

    def _clostest_template(self, templates, name):
        best = None, None
        best_similarity = 0
        md1 = self.typed_variables[name]
        for name2, template in templates.items():
            md2 = self.typed_variables[name2]
            similarity = md1.similarity(md2)
            if similarity > best_similarity:
                best = template, name2
                best_similarity = similarity
        return best
