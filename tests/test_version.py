

import pytest

from anemoi.inference.checkpoint.utils import Version
import pytest
from anemoi.inference.checkpoint.utils import Version

@pytest.mark.parametrize("version_str, expected_major, expected_minor, expected_patch", [
    ("1.0.0", 1, 0, 0),
    ("2.3.4", 2, 3, 4),
    ("0.1.2", 0, 1, 2),
    ("10.20.30", 10, 20, 30),
])
def test_version_value(version_str, expected_major, expected_minor, expected_patch):
    version = Version(version_str)
    assert version.major == expected_major
    assert version.minor == expected_minor
    assert version.patch == expected_patch

@pytest.mark.parametrize("version1, version2, expected_result", [
    ("1.0.0", "1.0.0", False),  # Same versions, expected result is True
    ("1.0.0", "2.0.0", False),  # version1 is lower than version2, expected result is False
    ("2.0.0", "1.0.0", True),  # version1 is higher than version2, expected result is True
    ("1.2.3", "1.2.4", False),  # version1 is lower than version2, expected result is False
    ("1.2.4", "1.2.3", True),  # version1 is higher than version2, expected result is True
    ("1.4.4", "1.2.3", True),  # version1 is higher than version2, expected result is True
    ("1.4.4", "1.2.32", True),  # version1 is higher than version2, expected result is True
    ("1.0.0", "1.0.1", False),  # version1 is lower than version2, expected result is False
    ("1.0.1", "1.0.0", True),  # version1 is higher than version2, expected result is True
    ("1.0.0", "1.1.0", False),  # version1 is lower than version2, expected result is False
    ("1.1.0", "1.0.0", True),  # version1 is higher than version2, expected result is True
    ("1.0.0", "2.0.1", False),  # version1 is lower than version2, expected result is False
    ("2.0.1", "1.0.0", True),  # version1 is higher than version2, expected result is True
])
def test_compare_versions_gt(version1, version2, expected_result):
    v1 = Version(version1)
    v2 = Version(version2)
    assert (v1 > v2) == expected_result

@pytest.mark.parametrize("version1, version2, expected_result", [
    ("1.0.0", "1.0.0", False),  # Same versions, expected result is False
    ("1.0.0", "2.0.0", True),  # version1 is lower than version2, expected result is True
    ("2.0.0", "1.0.0", False),  # version1 is higher than version2, expected result is False
    ("1.2.3", "1.2.4", True),  # version1 is lower than version2, expected result is True
    ("1.2.4", "1.2.3", False),  # version1 is higher than version2, expected result is False
    ("1.4.4", "1.2.3", False),  # version1 is higher than version2, expected result is False
    ("1.4.4", "1.2.32", False),  # version1 is higher than version2, expected result is False
    ("1.0.0", "1.0.1", True),  # version1 is lower than version2, expected result is True
    ("1.0.1", "1.0.0", False),  # version1 is higher than version2, expected result is False
    ("1.0.0", "1.1.0", True),  # version1 is lower than version2, expected result is True
    ("1.1.0", "1.0.0", False),  # version1 is higher than version2, expected result is False
    ("1.0.0", "2.0.1", True),  # version1 is lower than version2, expected result is True
    ("2.0.1", "1.0.0", False),  # version1 is higher than version2, expected result is False
])
def test_compare_versions_lt(version1, version2, expected_result):
    v1 = Version(version1)
    v2 = Version(version2)
    assert (v1 < v2) == expected_result

@pytest.mark.parametrize("version1, version2, expected_result", [
    ("1.0.0", "1.0.0", True),  # Same versions, expected result is True
    ("1.0.0", "2.0.0", False),  # Different versions, expected result is False
    ("2.0.0", "1.0.0", False),  # Different versions, expected result is False
    ("1.2.3", "1.2.3", True),  # Same versions, expected result is True
    ("1.2.4", "1.2.3", False),  # Different versions, expected result is False
    ("1.4.4", "1.2.3", False),  # Different versions, expected result is False
    ("1.4.4", "1.2.32", False),  # Different versions, expected result is False
    ("1.0.0", "1.0.1", False),  # Different versions, expected result is False
    ("1.0.1", "1.0.0", False),  # Different versions, expected result is False
    ("1.0.0", "1.1.0", False),  # Different versions, expected result is False
    ("1.1.0", "1.0.0", False),  # Different versions, expected result is False
    ("1.0.0", "2.0.1", False),  # Different versions, expected result is False
    ("2.0.1", "1.0.0", False),  # Different versions, expected result is False
])
def test_compare_versions_eq(version1, version2, expected_result):
    v1 = Version(version1)
    v2 = Version(version2)
    assert (v1 == v2) == expected_result


@pytest.mark.parametrize("version1, version2, expected_result", [
    ("1.0.0", "1.0.0", True),  # Same versions, expected result is True
    ("1.0.0", "2.0.0", False),  # Different versions, expected result is False
    ("2.0.0", "1.0.0", False),  # Different versions, expected result is False
    ("1.2.3", "1.2.3", True),  # Same versions, expected result is True
    ("1.2.4", "1.2.3", False),  # Different versions, expected result is False
    ("1.4.4", "1.2.3", False),  # Different versions, expected result is False
    ("1.4.4", "1.2.32", False),  # Different versions, expected result is False
    ("1.0.0", "1.0.1", False),  # Different versions, expected result is False
    ("1.0.1", "1.0.0", False),  # Different versions, expected result is False
    ("1.0.0", "1.1.0", False),  # Different versions, expected result is False
    ("1.1.0", "1.0.0", False),  # Different versions, expected result is False
    ("1.0.0", "2.0.1", False),  # Different versions, expected result is False
    ("2.0.1", "1.0.0", False),  # Different versions, expected result is False
    ("1.0.0", "1.0.0a", False),  # Different versions, expected result is False
    ("1.0.0a", "1.0.0", False),  # Different versions, expected result is False
    ("1.0.0a", "1.0.0b", False),  # Different versions, expected result is False
    ("1.0.0b", "1.0.0a", False),  # Different versions, expected result is False
    ("1.0.0a", "1.0.0a", True),  # Same versions, expected result is True
    ("1.0.0a", "1.0.0b", False),  # Different versions, expected result is False
    ("1.0.0b", "1.0.0a", False),  # Different versions, expected result is False
    ("0.4.5.dev37+g33820eb", "0.4.5.dev37+g33520eb", False),  # Different versions, expected result is False
])
def test_compare_versions_eq_with_str_patch(version1, version2, expected_result):
    v1 = Version(version1)
    v2 = Version(version2)
    assert (v1 == v2) == expected_result