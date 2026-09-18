# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import pytest

from anemoi.inference.outputs.printer import PrinterOutput

LOG = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "variables, expected_in_output, not_expected_in_output",
    [
        pytest.param(None, ["z_500"], ["cp", "tp"], id="none_input"),  # max_lines is 1, so only 1 should be printed
        pytest.param({"select": "cp"}, ["cp"], ["z_500", "tp"], id="select_single_string"),
        pytest.param({"select": ["z_500", "cp"]}, ["cp", "z_500"], ["tp"], id="select_list"),
        pytest.param({"drop": ["z", "cp"]}, ["tp", "z_500"], ["cp"], id="drop_list"),
        pytest.param({"drop": "cp"}, ["tp", "z_500"], ["cp"], id="drop_single_string"),
        pytest.param("all", ["z_500", "cp", "tp"], [], id="test_all_kwarg"),
    ],
)
def test_print_state_variable_inclusion(
    variables, expected_in_output, not_expected_in_output, basic_context, basic_metadata, basic_state, capsys
):
    output = PrinterOutput(basic_context, basic_metadata, max_lines=1, variables=variables)

    output.write_state(basic_state)
    output_str = capsys.readouterr()[0]
    LOG.warning(f"Maxine: {output_str}")

    for variable in expected_in_output:
        assert variable in output_str

    for variable in not_expected_in_output:
        assert variable not in output_str

    output.close()
