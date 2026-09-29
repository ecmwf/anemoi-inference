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
from anemoi.inference.outputs.printer import print_state

LOG = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "variables, expected_in_output, not_expected_in_output",
    [
        pytest.param(None, ["z_500"], ["cp", "2t"], id="none_input"),  # max_lines is 1, so only 1 should be printed
        pytest.param({"select": "cp"}, ["cp"], ["z_500", "2t"], id="select_single_string"),
        pytest.param({"select": ["z_500", "cp"]}, ["cp", "z_500"], ["2t"], id="select_list"),
        pytest.param({"drop": ["z", "cp"]}, ["2t", "z_500"], ["cp"], id="drop_list"),
        pytest.param({"drop": "cp"}, ["2t", "z_500"], ["cp"], id="drop_single_string"),
    ],
)
def test_print_state_variable_inclusion(variables, expected_in_output, not_expected_in_output, basic_state, capsys):
    print_state(basic_state, max_lines=1, variables=variables)
    output_str = capsys.readouterr()[0]

    for variable in expected_in_output:
        assert variable in output_str

    for variable in not_expected_in_output:
        assert variable not in output_str


@pytest.mark.parametrize(
    "variables, not_expected_in_output, max_lines",
    [
        pytest.param(None, ["z_500", "cp", "2t"], 0, id="none_variables_and_no_maxlines"),
        pytest.param(None, ["2t"], 2, id="none_variables_and_maxlines"),
        pytest.param({"select": ["z_500", "cp"]}, ["2t"], 1, id="set_variables_and_maxlines"),
        pytest.param({"drop": ["z_500"]}, ["z_500"], 1, id="drop_variables_and_maxlines"),
    ],
)
def test_print_state_max_lines(
    variables, not_expected_in_output, max_lines, basic_context, basic_metadata, basic_state, capsys
):
    output = PrinterOutput(basic_context, basic_metadata, max_lines=max_lines, variables=variables)

    output.write_state(basic_state)
    output_str = capsys.readouterr()[0]

    for variable in not_expected_in_output:
        assert variable not in output_str

    output.close()
