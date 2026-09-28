# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator
from pydantic import model_validator


class OutputVariableConfig(BaseModel):
    """Type for output variable configuration settings.

    Only one of `select` or `drop` have values, with the other being None. `select` indicates that only the variables provided
    should be written to the output. `drop` indicates that all variables EXCEPT those provided should be written to the output.

    To convert from the allowed types for configuration (str | list[str] | dict) to this type, use OutputVariableConfig.model_validate(variables).

    Attributes
    ----------
    select : list[str]
        Variables to include in the output. Defaults to None.
    drop : list[str]
        Variables to remove from the output. Defaults to None.
    """

    model_config = ConfigDict(extra="forbid")

    select: list[str] | None = None
    drop: list[str] | None = None

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
        if isinstance(value, str):
            return [value]
        else:
            return value

    @property
    def not_set(self) -> bool:
        return self.select is None and self.drop is None

    def skip(self, variable: str) -> bool:
        """Return True if the provided variable should be skipped, False otherwise."""
        skip_variable_select = self.select is not None and variable not in self.select
        skip_variable_drop = self.drop is not None and variable in self.drop

        return skip_variable_select or skip_variable_drop
