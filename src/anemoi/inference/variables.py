# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any
from typing import Dict


class Variables:

    def __init__(self, runner: Any) -> None:
        """Initialise the Variables object.

        Parameters
        ----------
        runner : Any
            The runner object containing the checkpoint.
        """
        self.runner = runner
        self.checkpoint = runner.checkpoint

    ###############################################################################
    @classmethod
    def default_runner_input_variables_include_exclude(cls) -> Dict[str, list[str]]:
        """Get include/exclude lists for default runner input variables.

        Returns
        -------
        dict
            Dictionary with 'include' and 'exclude' lists for default input variables.
        """
        return dict(
            include=["prognostic", "forcing"],
            exclude=["computed", "diagnostic"],
        )

    def default_input_variables(self) -> Any:
        """Select default input variables from the checkpoint.

        Returns
        -------
        Any
            The selected variables.
        """
        return self.checkpoint.select_variables(
            **self.default_runner_input_variables_include_exclude(),
        )

    def default_input_variables_and_mask(self) -> Any:
        """Select default input variables and masks from the checkpoint.

        Returns
        -------
        Any
            The selected variables and masks.
        """
        return self.checkpoint.select_variables_and_masks(
            **self.default_runner_input_variables_include_exclude(),
        )

    ###############################################################################
    @classmethod
    def retrieved_constant_forcings_variables_include_exclude(cls) -> Dict[str, list[str]]:
        """Get include/exclude lists for retrieved constant forcings variables.

        Returns
        -------
        dict
            Dictionary with 'include' and 'exclude' lists for constant forcings variables.
        """
        return dict(
            include=["constant+forcing"],
            exclude=["computed", "diagnostic", "prognostic"],
        )

    def retrieved_constant_forcings_variables(self) -> Any:
        """Select retrieved constant forcings variables from the checkpoint.

        Returns
        -------
        Any
            The selected constant forcings variables.
        """
        return self.checkpoint.select_variables(
            **self.retrieved_constant_forcings_variables_include_exclude(),
        )

    def retrieved_constant_forcings_variables_and_mask(self) -> Any:
        """Select retrieved constant forcings variables and masks from the checkpoint.

        Returns
        -------
        Any
            The selected constant forcings variables and masks.
        """
        return self.checkpoint.select_variables_and_masks(
            **self.retrieved_constant_forcings_variables_include_exclude(),
        )

    ###############################################################################
    @classmethod
    def retrieved_prognostic_variables_include_exclude(cls) -> Dict[str, list[str]]:
        """Get include/exclude lists for retrieved prognostic variables.

        Returns
        -------
        dict
            Dictionary with 'include' and 'exclude' lists for prognostic variables.
        """
        return dict(
            include=["prognostic"],
            exclude=["computed", "diagnostic", "forcing"],
        )

    def retrieved_prognostic_variables(self) -> Any:
        """Select retrieved prognostic variables from the checkpoint.

        Returns
        -------
        Any
            The selected prognostic variables.
        """
        return self.checkpoint.select_variables(
            **self.retrieved_prognostic_variables_include_exclude(),
        )

    def retrieved_prognostic_variables_and_mask(self) -> Any:
        """Select retrieved prognostic variables and masks from the checkpoint.

        Returns
        -------
        Any
            The selected prognostic variables and masks.
        """
        return self.checkpoint.select_variables_and_masks(
            **self.retrieved_prognostic_variables_include_exclude(),
        )

    ###############################################################################
    @classmethod
    def computed_constant_forcings_variables_include_exclude(cls) -> Dict[str, list[str]]:
        """Get include/exclude lists for computed constant forcings variables.

        Returns
        -------
        dict
            Dictionary with 'include' list for computed constant forcings variables.
        """
        return dict(
            include=["computed+constant+forcing"],
        )

    def computed_constant_forcings_variables(self) -> Any:
        """Select computed constant forcings variables from the checkpoint.

        Returns
        -------
        Any
            The selected computed constant forcings variables.
        """
        return self.checkpoint.select_variables(
            **self.computed_constant_forcings_variables_include_exclude(),
        )

    def computed_constant_forcings_variables_and_mask(self) -> Any:
        """Select computed constant forcings variables and masks from the checkpoint.

        Returns
        -------
        Any
            The selected computed constant forcings variables and masks.
        """
        return self.checkpoint.select_variables_and_masks(
            **self.computed_constant_forcings_variables_include_exclude(),
        )

    ###############################################################################
    @classmethod
    def retrieved_dynamic_forcings_variables_include_exclude(cls) -> Dict[str, list[str]]:
        """Get include/exclude lists for retrieved dynamic forcings variables.

        Returns
        -------
        dict
            Dictionary with 'include' and 'exclude' lists for dynamic forcings variables.
        """
        return dict(
            include=["forcing"],
            exclude=["computed", "constant"],
        )

    def retrieved_dynamic_forcings_variables(self) -> Any:
        """Select retrieved dynamic forcings variables from the checkpoint.

        Returns
        -------
        Any
            The selected dynamic forcings variables.
        """
        return self.checkpoint.select_variables(
            **self.retrieved_dynamic_forcings_variables_include_exclude(),
        )

    def retrieved_dynamic_forcings_variables_and_mask(self) -> Any:
        """Select retrieved dynamic forcings variables and masks from the checkpoint.

        Returns
        -------
        Any
            The selected dynamic forcings variables and masks.
        """
        return self.checkpoint.select_variables_and_masks(
            **self.retrieved_dynamic_forcings_variables_include_exclude(),
        )

    ###############################################################################

    @classmethod
    def input_types(cls) -> Dict[str, Dict[str, list[str]]]:
        """Get all input types and their include/exclude lists.

        Returns
        -------
        dict
            Dictionary mapping input type names to include/exclude dicts.
        """
        return {
            "default-input": cls.default_runner_input_variables_include_exclude(),  # For backwards compatibility
            "constant-forcings": cls.retrieved_constant_forcings_variables_include_exclude(),
            "prognostics": cls.retrieved_prognostic_variables_include_exclude(),
            "dynamic-forcings": cls.retrieved_dynamic_forcings_variables_include_exclude(),
        }

    @classmethod
    def input_type_to_include_exclude(cls, input_type: str) -> Dict[str, list[str]]:
        """Get the include/exclude dict for a given input type.

        Parameters
        ----------
        input_type : str
            The name of the input type.

        Returns
        -------
        dict
            The include/exclude dict for the input type.
        """
        return cls.input_types()[input_type]
