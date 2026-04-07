from typing import Any

import pytest
import yaml
from pytest import MonkeyPatch

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.config.utils import multi_datasets_config
from anemoi.inference.testing import files_for_tests


def test_load_config() -> None:
    """Test loading the configuration"""
    RunConfiguration.load(files_for_tests("unit/configs/simple.yaml"))


def test_interpolation(monkeypatch: MonkeyPatch) -> None:
    """Test loading the configuration with some OmegaConf interpolations"""
    monkeypatch.setenv("TEST", "foo")
    config = RunConfiguration.load(files_for_tests("unit/configs/interpolation.yaml"))
    assert config.name == "foo"
    assert config.name == config.description


def test_config_dotlist_override() -> None:
    """Test overriding the configuration with some dotlist parameters"""
    config = RunConfiguration.load(
        files_for_tests("unit/configs/mwd.yaml"),
        overrides=[
            "runner=testing",
            "device=cpu",
            "input=dummy",
            "post_processors.0.backward_transform_filter=test",
        ],
    )
    assert config.post_processors is not None and len(config.post_processors) == 1
    assert isinstance(config.post_processors[0], dict)
    assert config.post_processors[0]["backward_transform_filter"] == "test"


def test_config_dotlist_override_append() -> None:
    """Test overriding with an additional parameter"""
    config = RunConfiguration.load(
        files_for_tests("unit/configs/mwd.yaml"),
        overrides=[
            "runner=testing",
            "device=cpu",
            "input=dummy",
            "post_processors.1.backward_transform_filter=test",
        ],
    )
    assert config.post_processors is not None and len(config.post_processors) == 2
    assert isinstance(config.post_processors[1], dict)
    assert config.post_processors[1]["backward_transform_filter"] == "test"


def test_config_dotlist_override_append_end() -> None:
    """Test overriding with an additional parameter"""
    config = RunConfiguration.load(
        files_for_tests("unit/configs/simple.yaml"),
        overrides=[
            "pre_processors=[]",
            "pre_processors.0=test",
        ],
    )
    assert isinstance(config.pre_processors, list) and len(config.pre_processors) == 1
    assert config.pre_processors[0] == "test"


def test_config_dotlist_override_add_new_non_empty_dict() -> None:
    """Test overriding with an additional parameter"""
    config = RunConfiguration.load(
        files_for_tests("unit/configs/simple.yaml"),
        overrides=["input={grib: test}"],
    )
    assert isinstance(config.input, dict) and "grib" in config.input
    assert config.input["grib"] == "test"


def test_config_dotlist_override_add_new_non_empty_list() -> None:
    """Test overriding with an additional parameter"""
    config = RunConfiguration.load(
        files_for_tests("unit/configs/simple.yaml"),
        overrides=["pre_processors=[test]"],
    )
    assert isinstance(config.pre_processors, list) and len(config.pre_processors) == 1
    assert config.pre_processors[0] == "test"


def test_config_dotlist_override_index_error() -> None:
    """Test failing with index error in dotlist argument"""
    with pytest.raises(IndexError):
        RunConfiguration.load(
            files_for_tests("unit/configs/mwd.yaml"),
            overrides=[
                "runner=testing",
                "device=cpu",
                "input=dummy",
                "post_processors.2.backward_transform_filter=test",
            ],
        )


def test_config_patch_metadata_file(tmp_path) -> None:
    """Test loading the configuration with patch metadata from a file"""

    patch_metadata = {"dataset": {"variables_metadata": {"2t": "some_patch"}}}

    patch = tmp_path / "patch.yaml"
    patch.write_text(yaml.dump(patch_metadata))
    config = RunConfiguration.load(
        files_for_tests("unit/configs/simple.yaml"),
        overrides=[f"patch_metadata={patch}"],
    )
    assert isinstance(config.patch_metadata, dict)
    assert config.patch_metadata["dataset"]["variables_metadata"]["2t"] == "some_patch"


@pytest.mark.parametrize(
    "config, dataset_name, datasets, expected, strict",
    [
        # Single dataset present in config -> return its value
        ({"era5": {"grib": "out.grib"}}, "era5", ["era5"], {"grib": "out.grib"}, True),
        # Multi-dataset config -> return matching dataset
        (
            {"era5": {"grib": "out-era5.grib"}, "cerra": {"netcdf": "out-cerra.nc"}},
            "era5",
            ["era5", "cerra"],
            {"grib": "out-era5.grib"},
            True,
        ),
        (
            {"era5": {"grib": "out-era5.grib"}, "cerra": {"netcdf": "out-cerra.nc"}},
            "cerra",
            ["era5", "cerra"],
            {"netcdf": "out-cerra.nc"},
            True,
        ),
        # Config with a single key -> return as-is
        ({"output": "file.grib"}, "era5", ["era5", "cerra"], {"output": "file.grib"}, True),
        # Non-dict config -> return as-is
        ("some_string", "era5", ["era5"], "some_string", True),
        ([1, 2, 3], "era5", ["era5"], [1, 2, 3], True),
        # strict=False with multi-key dict where dataset_name is missing -> return as-is
        ({"a": 1, "b": 2}, "era5", ["era5"], {"a": 1, "b": 2}, False),
    ],
)
def test_multi_datasets_config(
    config: Any, dataset_name: str, datasets: list[str], expected: Any, strict: bool
) -> None:
    """Test the multi_datasets_config function with various inputs and expected outputs."""
    result = multi_datasets_config(config, dataset_name, datasets, strict=strict)
    assert result == expected


@pytest.mark.parametrize(
    "config, dataset_name, datasets, strict",
    [
        # Multi-key dict where dataset_name is missing
        ({"a": 1, "b": 2}, "era5", ["era5"], True),
        # Typo in one of the dataset names
        ({"era5": {"grib": "out.grib"}, "berra": "x"}, "era5", ["era5", "cerra"], True),
        # Only one of the datasets present
        ({"era5": {"grib": "out.grib"}}, "era5", ["era5", "cerra"], True),
    ],
)
def test_multi_datasets_config_errors(config: Any, dataset_name: str, datasets: list[str], strict: bool) -> None:
    """Test that multi_datasets_config raises AssertionError for invalid inputs."""
    with pytest.raises(AssertionError):
        multi_datasets_config(config, dataset_name, datasets, strict=strict)
