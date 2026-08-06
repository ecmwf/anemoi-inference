# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import pytest
from anemoi.transform.variables import Variable

from anemoi.inference.output import Output
from anemoi.inference.output import _parse_levels


@pytest.mark.parametrize(
    "spec, expected",
    [
        (100, [100]),
        ([50, 51], [50, 51]),
        ("50/to/53", [50, 51, 52, 53]),
        ("50/to/56/by/2", [50, 52, 54, 56]),
        ("53/to/50", [53, 52, 51, 50]),
    ],
)
def test_parse_levels(spec, expected):
    assert _parse_levels(spec) == expected


def test_parse_levels_range_length():
    assert len(_parse_levels("50/to/137")) == 88


class _Concrete(Output):
    def open(self, state):  # pragma: no cover - not exercised
        pass

    def write_step(self, state):  # pragma: no cover - not exercised
        pass

    def write_initial_step(self, state):  # pragma: no cover - not exercised
        pass


class _Var:
    def __init__(self, param, level):
        self.param = param
        self.level = level


def _make_output(variables):
    typed_variables = {
        "u_100": _Var("u", 100),
        "u_200": _Var("u", 200),
        "v_50": _Var("v", 50),
        "v_137": _Var("v", 137),
        "t_137": _Var("t", 137),
        "t_50": _Var("t", 50),
        "tp": _Var("tp", None),
    }
    context = SimpleNamespace(reference_date=None, typed_variables={})
    metadata = SimpleNamespace(dataset_name="test", typed_variables=typed_variables)
    return _Concrete(context, metadata, variables=variables)


def _make_output_with_real_variables(variables):
    typed_variables = {
        "u_100": Variable.from_dict(
            "u_100",
            {
                "mars": {
                    "param": "u",
                    "levtype": "pl",
                    "levelist": 100,
                }
            },
        ),
        "u_200": Variable.from_dict(
            "u_200",
            {
                "mars": {
                    "param": "u",
                    "levtype": "pl",
                    "levelist": 200,
                }
            },
        ),
    }
    context = SimpleNamespace(reference_date=None, typed_variables={})
    metadata = SimpleNamespace(dataset_name="test", typed_variables=typed_variables)
    return _Concrete(context, metadata, variables=variables)


def test_skip_variable_none_keeps_everything():
    out = _make_output(None)
    assert out.skip_variable("anything") is False


def test_skip_variable_plain_names_backward_compatible():
    out = _make_output(["t_137", "tp"])
    assert out.skip_variable("t_137") is False
    assert out.skip_variable("tp") is False
    assert out.skip_variable("t_50") is True


def test_skip_variable_param_level_range():
    out = _make_output([{"param": "u", "level": "50/to/137"}])
    assert out.skip_variable("u_100") is False
    assert out.skip_variable("u_200") is True  # outside the range


def test_skip_variable_param_level_list():
    out = _make_output([{"param": "v", "level": [50, 51]}])
    assert out.skip_variable("v_50") is False
    assert out.skip_variable("v_137") is True


def test_skip_variable_param_only_matches_all_levels():
    out = _make_output([{"param": "tp"}])
    assert out.skip_variable("tp") is False


def test_skip_variable_mixed_selectors_and_names():
    out = _make_output(
        [
            {"param": "u", "level": "50/to/137"},
            {"param": "v", "level": [50, 51]},
            "t_137",
        ]
    )
    assert out.skip_variable("u_100") is False
    assert out.skip_variable("v_50") is False
    assert out.skip_variable("t_137") is False
    assert out.skip_variable("t_50") is True
    assert out.skip_variable("u_200") is True


def test_skip_variable_param_level_with_real_variable_metadata():
    out = _make_output_with_real_variables([{"param": "u", "level": "50/to/137"}])
    assert out.skip_variable("u_100") is False
    assert out.skip_variable("u_200") is True
