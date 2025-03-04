# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import json
import logging
from abc import abstractmethod

from earthkit.data.utils.dates import to_datetime

from ..grib.encoding import grib_keys
from ..grib.templates.manager import TemplateManager
from ..output import Output

LOG = logging.getLogger(__name__)


class HindcastOutput:

    def __init__(self, reference_year):
        self.reference_year = reference_year

    def __call__(self, values, template, keys):

        if "date" not in keys:
            assert template.metadata("hdate", default=None) is None, template
            date = template.metadata("date")
        else:
            date = keys.pop("date")

        for k in ("date", "hdate"):
            keys.pop(k, None)

        keys["edition"] = 1
        keys["localDefinitionNumber"] = 30
        keys["dataDate"] = int(to_datetime(date).strftime("%Y%m%d"))
        keys["referenceDate"] = int(to_datetime(date).replace(year=self.reference_year).strftime("%Y%m%d"))

        return values, template, keys


MODIFIERS = dict(hindcast=HindcastOutput)


def modifier_factory(modifiers):

    if modifiers is None:
        return []

    if not isinstance(modifiers, list):
        modifiers = [modifiers]

    result = []
    for modifier in modifiers:
        assert isinstance(modifier, dict), modifier
        assert len(modifier) == 1, modifier

        klass = list(modifier.keys())[0]
        result.append(MODIFIERS[klass](**modifier[klass]))

    return result


class GribOutput(Output):
    """Handles grib"""

    def __init__(
        self,
        context,
        *,
        encoding=None,
        templates=None,
        grib1_keys=None,
        grib2_keys=None,
        modifiers=None,
        output_frequency=None,
        write_initial_state=None,
        variables=None,
    ):
        super().__init__(context, output_frequency=output_frequency, write_initial_state=write_initial_state)
        self._first = True
        self.typed_variables = self.checkpoint.typed_variables
        self.encoding = encoding if encoding is not None else {}
        self.grib1_keys = grib1_keys if grib1_keys is not None else {}
        self.grib2_keys = grib2_keys if grib2_keys is not None else {}

        self.modifiers = modifier_factory(modifiers)
        self.variables = variables

        self.ensemble = False
        for d in (self.grib1_keys, self.grib2_keys, self.encoding):
            if "eps" in d:
                self.ensemble = d["eps"]
                break
            if d.get("type") in ("pf", "cf"):
                self.ensemble = True
                break

        self.template_manager = TemplateManager(self, templates)

        self.ensemble = False
        for d in (self.grib1_keys, self.grib2_keys, self.encoding):
            if "eps" in d:
                self.ensemble = d["eps"]
                break
            if d.get("type") in ("pf", "cf"):
                self.ensemble = True
                break

        self.template_manager = TemplateManager(self, templates)

    def write_initial_step(self, state):
        # We trust the GribInput class to provide the templates
        # matching the input state

        state = state.copy()

        self.reference_date = state["date"]
        state.setdefault("step", datetime.timedelta(0))

        out_vars = self.variables if self.variables is not None else state["fields"].keys()

        for name in out_vars:
            variable = self.typed_variables[name]

            if variable.is_computed_forcing:
                continue

            template = self.template(state, name)
            if template is None:
                # We can currently only write grib output if we have a grib input
                raise ValueError(
                    "GRIB output only works if the input is GRIB (for now). Set `write_initial_step` to `false`."
                )

        return self.write_step(state)

    def write_step(self, state):

        reference_date = self.reference_date or self.context.reference_date
        step = state["step"]
        previous_step = state.get("previous_step")
        start_steps = state.get("start_steps", {})

        out_vars = self.variables if self.variables is not None else state["fields"].keys()
        for name in out_vars:
            values = state["fields"][name]
            keys = {}

            variable = self.typed_variables[name]

            if variable.is_computed_forcing:
                continue

            param = variable.grib_keys.get("param", name)

            template = self.template(state, name)

            keys.update(self.encoding)

            keys = grib_keys(
                values=values,
                template=template,
                date=int(reference_date.strftime("%Y%m%d")),
                time=reference_date.hour * 100,
                step=step,
                param=param,
                variable=variable,
                ensemble=self.ensemble,
                keys=keys,
                grib1_keys=self.grib1_keys,
                grib2_keys=self.grib2_keys,
                previous_step=previous_step,
                start_steps=start_steps,
            )

            for modifier in self.modifiers:
                values, template, keys = modifier(values, template, keys)

            if LOG.isEnabledFor(logging.DEBUG):
                LOG.info("Encoding GRIB %s\n%s", template, json.dumps(keys, indent=4))

            try:
                self.write_message(values, template=template, **keys)
            except Exception:
                LOG.error("Error writing field %s", name)
                LOG.error("Template: %s", template)
                LOG.error("Keys:\n%s", json.dumps(keys, indent=4, default=str))
                raise

    @abstractmethod
    def write_message(self, message, *args, **kwargs):
        pass

    def template(self, state, name):

        if self.template_manager is None:
            self.template_manager = TemplateManager(self, self.templates)

        return self.template_manager.template(name, state)

    def template_lookup(self, name):
        return self.encoding
