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
        (None, ["cp", "tp", "z_500"]),
        (["cp"], ["cp"]),
        (["z_500", "cp"], ["cp", "z_500"]),
        ("cp", ["cp"]),
        ({"select": "cp"}, ["cp"]),
        ({"select": ["z_500", "cp"]}, ["cp", "z_500"]),
        ({"drop": ["z", "cp"]}, ["tp", "z_500"]),
        ({"drop": "cp"}, ["tp", "z_500"]),
        ({"drop": []}, ["cp", "tp", "z_500"]),
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
