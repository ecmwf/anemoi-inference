# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import fnmatch
from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator
from pydantic import model_validator

if TYPE_CHECKING:
    from anemoi.transform.variables import Variable

"""A glob pattern on the variable name, or a mapping of MARS keys that must all match."""
NameOrMarsMap = str | dict[str, Any]


class OutputVariableConfig(BaseModel):
    """Type for output variable configuration settings.

    Only one of `select` or `drop` have values, with the other being None. `select` indicates that only the variables provided
    should be written to the output. `drop` indicates that all variables EXCEPT those provided should be written to the output.
    Entries are either shell-style glob patterns (``*``, ``?``, ``[...]``) matched against the variable name, or mappings
    of MARS keys (for example ``{"levtype": "pl"}``) matched against the variable metadata. A literal name matches only
    itself. All keys in a mapping must match; separate entries are alternatives.

    To convert from the allowed types for configuration (str | list[str] | dict) to this type, use OutputVariableConfig.model_validate(variables).

    Attributes
    ----------
    select : list[NameOrMarsMap]
        Selectors for variables to include in the output. Defaults to None.
    drop : list[NameOrMarsMap]
        Selectors for variables to remove from the output. Defaults to None.
    """

    model_config = ConfigDict(extra="forbid")

    select: list[NameOrMarsMap] | None = None
    drop: list[NameOrMarsMap] | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_config(cls, variables: Any) -> Any:
        if variables is None:
            return {}
        if isinstance(variables, str):
            return {"select": [variables]}
        if isinstance(variables, list):
            return {"select": variables}
        return variables

    @model_validator(mode="after")
    def check_either_or(self) -> "OutputVariableConfig":
        if self.select is not None and self.drop is not None:
            raise ValueError("Only one of `select` or `drop` can be set.")
        return self

    @field_validator("select", "drop", mode="before")
    @classmethod
    def ensure_list(cls, value: Any) -> Any:
        if isinstance(value, (str, dict)):
            return [value]
        else:
            return value

    @property
    def not_set(self) -> bool:
        return self.select is None and self.drop is None

    def skip(self, variable: str, typed_variable: "Variable | None" = None) -> bool:
        """Return True if the provided variable should be skipped, False otherwise.

        Parameters
        ----------
        variable : str
            The variable name.
        typed_variable : Variable, optional
            Metadata for the variable. Without it, MARS key selectors never match.

        Returns
        -------
        bool
            True if the variable should be skipped (based on inputs), false if the variable should be included.
        """
        if self.select is not None:
            return not any(self._matches(s, variable, typed_variable) for s in self.select)
        if self.drop is not None:
            return any(self._matches(s, variable, typed_variable) for s in self.drop)
        return False

    @staticmethod
    def _matches(selector: NameOrMarsMap, variable: str, typed_variable: "Variable | None") -> bool:
        """Compare the selector configuration, which is a glob pattern OR a mapping of MARS key: glob pattern, to the input variable or typed_variable."""
        if isinstance(selector, str):
            return fnmatch.fnmatchcase(variable, selector)
        if typed_variable is None:
            return False
        keys = typed_variable.grib_keys
        return all(key in keys and fnmatch.fnmatchcase(str(keys[key]), str(value)) for key, value in selector.items())
