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

from ..inputs import create_input
from ..output import Output

LOG = logging.getLogger(__name__)


class GribOutput(Output):
    """
    Handles grib
    """

    def __init__(self, context, *, encoding=None, templates=None):
        super().__init__(context)
        self._first = True
        self.typed_variables = self.checkpoint.typed_variables
        self.quiet = set()
        self.encoding = encoding if encoding is not None else {}
        self.edition = self.encoding.get("edition")
        self.templates = templates
        self._template_cache = None
        self._template_source = None
        self._template_date = None
        self._template_reuse = None

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

            keys = {}
            self.set_forecast(keys, None, 0)
            self.set_other_keys(keys, variable)
            self.write_message(None, template=template, **keys)

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

            def _clostest_template(name):
                best = None, None
                best_similarity = 0
                md1 = self.typed_variables[name]
                for name2, template in self.template(state, None).items():
                    md2 = self.typed_variables[name2]
                    similarity = md1.similarity(md2)
                    if similarity > best_similarity:
                        best = template, name2
                        best_similarity = similarity
                return best

            template = self.template(state, name)
            if template is None:
                if name not in self.quiet:
                    LOG.warning("No GRIB template found for `%s`. This may lead to unexpected results.", name)

                grib_keys = variable.grib_keys.copy()
                for key in ("class", "type", "stream", "expver", "date", "time", "step"):
                    grib_keys.pop(key, None)

                template, name2 = _clostest_template(name)

                if name not in self.quiet:
                    if name2 is not None:
                        LOG.warning("Using template for `%s` with keys %s", name2, grib_keys)
                    else:
                        LOG.warning("Using %s", grib_keys)
                    self.quiet.add(name)

                keys.update(grib_keys)

            keys.update(
                date=reference_date.strftime("%Y-%m-%d"),
                time=reference_date.hour,
                step=step,
                param=param,
            )

            self.set_forecast(keys, reference_date, step)
            self.set_other_keys(keys, variable)
            if variable.is_accumulation:
                self.set_accumulation(keys, 0, step, template=template)

            if self.edition is not None:
                keys["edition"] = self.edition

            if LOG.isEnabledFor(logging.DEBUG):
                LOG.debug("Encoding GRIB %s\n%s", template, json.dumps(keys, indent=4))

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

    def set_forecast(self, keys, reference_date, step):
        if reference_date is not None:
            keys["date"] = reference_date.strftime("%Y-%m-%d")
            keys["time"] = reference_date.hour

        keys["step"] = step
        keys["type"] = "fc"

        if self.edition == 2:
            keys["typeOfProcessedData"] = 1

        grib_keys = self.encoding.get("forecast", {})
        keys.update(grib_keys)

    def set_accumulation(self, keys, start, end, template):

        edition = self.edition
        if edition is None and template is not None:
            edition = template.metadata("edition")
            centre = template.metadata("centre")

        if edition == 2:
            keys["typeOfStatisticalProcessing"] = 1
            keys["startStep"] = start
            keys["endStep"] = end
            keys["stepType"] = "accum"
            keys["step"] = end
        elif edition == 1:
            # This is ecmwf specific :-(
            if centre == "ecmf":
                keys["timeRangeIndicator"] = 1
                keys["step"] = end
            else:
                keys["timeRangeIndicator"] = 4
                keys["startStep"] = start
                keys["endStep"] = end
                keys["stepType"] = "accum"
        else:
            # We don't know the edition
            keys["startStep"] = start
            keys["endStep"] = end
            keys["stepType"] = "accum"

        grib_keys = self.encoding.get("accumulation", {})
        keys.update(grib_keys)

    def set_other_keys(self, keys, variable):
        grib_keys = {k: v for k, v in self.encoding.items() if not isinstance(v, (dict, list))}
        keys.update(grib_keys)

        per_variable = self.encoding.get("per_variable", {})
        per_variable = per_variable.get(variable.name, {})
        keys.update(per_variable)

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

        date = self._template_date if self._template_date is not None else state["date"]
        field = self._template_source.retrieve(variables=[name], dates=[date])[0]

        if self._template_reuse:
            self._template_cache[None] = field
        else:
            self._template_cache[name] = field

        return field
