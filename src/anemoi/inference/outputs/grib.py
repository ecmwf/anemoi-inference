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
from typing import Any

from earthkit.data.utils.dates import to_datetime

from anemoi.inference.types import FloatArray
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State

from ..grib.encoding import grib_keys
from ..grib.templates.manager import TemplateManager
from ..output import Output

LOG = logging.getLogger(__name__)


class HindcastOutput:
    """Hindcast output class."""

    def __init__(self, reference_year: int) -> None:
        """Initialize the HindcastOutput object.

        Parameters
        ----------
        reference_year : int
            The reference year.
        """

        self.reference_year = reference_year

    def __call__(self, values: FloatArray, template: object, keys: dict) -> tuple:
        """Call the HindcastOutput object.

        Parameters
        ----------
        values : FloatArray
            The values array.
        template : object
            The template object.
        keys : dict
            The keys dictionary.

        Returns
        -------
        tuple
            The modified values, template, and keys.
        """

        if "date" not in keys:
            assert template.metadata("hdate", default=None) is None, template
            date = template.metadata("date")
        else:
            date = keys.pop("date")

        for k in ("date", "hdate", "eps", "productDefinitionTemplateNumber"):
            keys.pop(k, None)

        keys["localDefinitionNumber"] = 30
        keys["dataDate"] = int(to_datetime(date).strftime("%Y%m%d"))
        keys["referenceDate"] = int(to_datetime(date).replace(year=self.reference_year).strftime("%Y%m%d"))

        return values, template, keys


MODIFIERS = dict(hindcast=HindcastOutput)


def modifier_factory(modifiers: list) -> list:
    """Create a list of modifier instances.

    Parameters
    ----------
    modifiers : list
        A list of modifier configurations.

    Returns
    -------
    list
        A list of modifier instances.
    """

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


class BaseGribOutput(Output):
    """Handles grib."""

    def __init__(
        self,
        context: dict,
        post_processors: list[ProcessorConfig] | None = None,
        *,
        encoding: dict[str, Any] | None = None,
        templates: list[str] | str | None = None,
        grib1_keys: dict[str, Any] | None = None,
        grib2_keys: dict[str, Any] | None = None,
        modifiers: list[str] | None = None,
        variables: list[str] | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
    ) -> None:
        """Initialize the GribOutput object.

        Parameters
        ----------
        context : dict
            The context dictionary.
        post_processors : Optional[List[ProcessorConfig]] = None
            Post-processors to apply to the input
        encoding : dict, optional
            The encoding dictionary, by default None.
        templates : list or str, optional
            The templates list or string, by default None.
        grib1_keys : dict, optional
            The grib1 keys dictionary, by default None.
        grib2_keys : dict, optional
            The grib2 keys dictionary, by default None.
        modifiers : list, optional
            The list of modifiers, by default None.
        output_frequency : int, optional
            The frequency of output, by default None.
        write_initial_state : bool, optional
            Whether to write the initial state, by default None.
        variables : list, optional
            The list of variables, by default None.
        """

        super().__init__(
            context,
            variables=variables,
            post_processors=post_processors,
            output_frequency=output_frequency,
            write_initial_state=write_initial_state,
        )
        self._first = True

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

    def write_initial_state(self, state: State) -> None:
        """Write the initial step of the state.

        Parameters
        ----------
        state : State
            The state object.
        """
        # We trust the GribInput class to provide the templates
        # matching the input state

        if not self.write_step_zero:
            return

        state = state.copy()

        self.reference_date = state["date"]
        state.setdefault("step", datetime.timedelta(0))

        for name in state["fields"].keys():
            if self.skip_variable(name):
                continue

            variable = self.typed_variables[name]

            if variable.is_computed_forcing:
                continue

            template = self.template(state, name)
            if template is None:
                # grib output only reliably works when we have grib input, everything else relies on external templates
                raise ValueError(
                    f"No grib template found for initial state param `{name}`. Try setting `write_initial_state` to `false`."
                )

        return self.write_step(self.post_process(state))

    def write_step(self, state: State) -> None:
        """Write a step of the state.

        Parameters
        ----------
        state : State
            The state object.
        """

        reference_date = self.reference_date or self.context.reference_date
        step = state["step"]
        previous_step = state.get("previous_step")
        start_steps = state.get("start_steps", {})

        for name in state["fields"].keys():
            if self.skip_variable(name):
                continue

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
    def write_message(self, message: FloatArray, *args: Any, **kwargs: Any) -> None:
        """Write a message to the grib file.

        Parameters
        ----------
        message : FloatArray
            The message array.
        *args : Any
            Additional arguments.
        **kwargs : Any
            Additional keyword arguments.
        """
        pass

    def template(self, state: State, name: str) -> object:
        """Get the template for a variable.

        Parameters
        ----------
        state : State
            The state object.
        name : str
            The variable name.

        Returns
        -------
        object
            The template object.
        """

        if self.template_manager is None:
            self.template_manager = TemplateManager(self, self.templates)

        return self.template_manager.template(name, state, self.typed_variables)

    def template_lookup(self, name: str) -> dict:
        """Lookup the template for a variable.

        Parameters
        ----------
        name : str
            The variable name.

        Returns
        -------
        dict
            The template dictionary.
        """
        return self.encoding
