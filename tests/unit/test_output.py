# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest

from anemoi.inference.outputs.raw import RawOutput


@pytest.mark.parametrize(
    "input_variables, expected_written",
    [
        pytest.param(None, ["cp", "tp", "z_500"], id="none_input"),
        pytest.param(["cp"], ["cp"], id="single_list_input"),
        pytest.param(["z_500", "cp"], ["cp", "z_500"], id="list_input"),
        pytest.param("cp", ["cp"], id="single_string_input"),
        pytest.param({"select": "cp"}, ["cp"], id="select_string"),
        pytest.param({"select": ["z_500", "cp"]}, ["cp", "z_500"], id="select_list"),
        pytest.param({"drop": ["z", "cp"]}, ["tp", "z_500"], id="drop_list"),
        pytest.param({"drop": "cp"}, ["tp", "z_500"], id="drop_string"),
        pytest.param({"drop": []}, ["cp", "tp", "z_500"], id="drop_empty_list"),
        pytest.param({"select": []}, [], id="select_empty_list"),
    ],
)
def test_output_variables(
    input_variables, expected_written, tmp_path, mocker, basic_context, basic_metadata, basic_state
):
    writer = mocker.patch("anemoi.inference.outputs.raw.np.savez_compressed")
    output = RawOutput(basic_context, basic_metadata, dir=str(tmp_path), variables=input_variables)
    output.write_step(basic_state)
    written = writer.call_args.kwargs

    assert sorted(key.removeprefix("field_") for key in written if key.startswith("field_")) == expected_written


@pytest.mark.parametrize(
    "input_variables",
    [
        pytest.param({"": ["cp"]}, id="empty_dict_key"),
        pytest.param({"rop": ["cp"]}, id="typo"),
        pytest.param({"select": ["cp"], "drop": ["z_500"]}, id="select_and_drop"),
    ],
)
def test_output_variables_failure(input_variables, tmp_path, mocker, basic_context, basic_metadata, basic_state):
    with pytest.raises(ValueError):
        output = RawOutput(basic_context, basic_metadata, dir=str(tmp_path), variables=input_variables)
        output.write_step(basic_state)
