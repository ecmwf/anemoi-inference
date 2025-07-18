# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


class Variables:

    def __init__(self, runner) -> None:
        self.runner = runner
        self.checkpoint = runner.checkpoint

    ###############################################################################
    @classmethod
    def default_runner_input_variables_include_exclude(cls):
        return dict(
            include=["prognostic", "forcing"],
            exclude=["computed", "diagnostic"],
        )

    def default_input_variables(self):
        return self.checkpoint.select_variables(
            **self.default_runner_input_variables_include_exclude(),
        )

    def default_input_variables_and_mask(self):
        return self.checkpoint.select_variables_and_masks(
            **self.default_runner_input_variables_include_exclude(),
        )

    ###############################################################################
    @classmethod
    def retrieved_constant_forcings_variables_include_exclude(cls):
        return dict(
            include=["constant+forcing"],
            exclude=["computed", "diagnostic", "prognostic"],
        )

    def retrieved_constant_forcings_variables(self):
        return self.checkpoint.select_variables(
            **self.retrieved_constant_forcings_variables_include_exclude(),
        )

    def retrieved_constant_forcings_variables_and_mask(self):
        return self.checkpoint.select_variables_and_masks(
            **self.retrieved_constant_forcings_variables_include_exclude(),
        )

    ###############################################################################
    @classmethod
    def retrieved_prognostic_variables_include_exclude(cls):
        return dict(
            include=["prognostic"],
            exclude=["computed", "diagnostic", "forcing"],
        )

    def retrieved_prognostic_variables(self):
        return self.checkpoint.select_variables(
            **self.retrieved_prognostic_variables_include_exclude(),
        )

    def retrieved_prognostic_variables_and_mask(self):
        return self.checkpoint.select_variables_and_masks(
            **self.retrieved_prognostic_variables_include_exclude(),
        )

    ###############################################################################
    @classmethod
    def computed_constant_forcings_variables_include_exclude(cls):
        return dict(
            include=["computed+constant+forcing"],
        )

    def computed_constant_forcings_variables(self):
        return self.checkpoint.select_variables(
            **self.computed_constant_forcings_variables_include_exclude(),
        )

    def computed_constant_forcings_variables_and_mask(self):
        return self.checkpoint.select_variables_and_masks(
            **self.computed_constant_forcings_variables_include_exclude(),
        )

    ###############################################################################
    @classmethod
    def retrieved_dynamic_forcings_variables_include_exclude(cls):
        return dict(
            include=["forcing"],
            exclude=["computed", "constant"],
        )

    def retrieved_dynamic_forcings_variables(self):
        return self.checkpoint.select_variables(
            **self.retrieved_dynamic_forcings_variables_include_exclude(),
        )

    def retrieved_dynamic_forcings_variables_and_mask(self):
        return self.checkpoint.select_variables_and_masks(
            **self.retrieved_dynamic_forcings_variables_include_exclude(),
        )
