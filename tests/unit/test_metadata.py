# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import numpy as np
import pytest

from anemoi.inference.metadata import InferenceMetadata
from anemoi.inference.metadata import new_keys_of_patch
from anemoi.inference.testing.mock_checkpoint import mock_load_metadata
from anemoi.inference.testing.variables import w_100
from anemoi.inference.testing.variables import z
from anemoi.inference.testing.variables import z_100_renamed
from anemoi.inference.testing.variables import z_renamed


@pytest.mark.parametrize(
    "initial, patch, expected_new_keys",
    [
        # updating an existing leaf does not create any new key
        (
            {"config": {"dataloader": {"dataset": "abc"}}},
            {"config": {"dataloader": {"dataset": "xyz"}}},
            [],
        ),
        # a new nested subtree is reported by its top-most new key only
        (
            {"config": {"dataloader": {"dataset": "abc"}}},
            {"config": {"dataloader": {"something": {"else": "123"}}}},
            ["config.dataloader.something"],
        ),
        # a new leaf alongside an existing one is reported on its own
        (
            {"a": {"b": 1, "c": 2}},
            {"a": {"c": 3, "d": 4}},
            ["a.d"],
        ),
        # a brand-new top-level key is reported
        (
            {"a": 1},
            {"b": 2},
            ["b"],
        ),
        # replacing a list by a dict is not a new key
        (
            {"config": {"dataloader": [{"dataset": "abc"}]}},
            {"config": {"dataloader": {"cutout": []}}},
            [],
        ),
    ],
)
def test_new_keys_of_patch(initial, patch, expected_new_keys):
    assert new_keys_of_patch(initial, patch) == expected_new_keys


def test_constant_fields_patch(atmos_core):
    fields = ["z", "sdor", "slor", "lsm"]
    metadata = InferenceMetadata(atmos_core.with_overrides({"dataset": {"constant_fields": fields}}))

    variables_metadata = metadata.dataset_view.data.variables_metadata()
    assert all(variables_metadata[name]["constant_in_time"] for name in fields)
    assert set(fields) <= set(metadata.select_variables(include=["constant"]))
    # the patch did not touch the original
    assert "constant_fields" not in atmos_core["dataset"]


def test_flags(atmos_core, multi_core):
    atmos, multi = InferenceMetadata(atmos_core), InferenceMetadata(multi_core)

    assert not atmos.multi_dataset and multi.multi_dataset
    assert multi.dataset_name == "data"

    # the richer batch layout starts with metadata version 3.0
    assert not atmos.richer_batch  # no version
    assert not multi.richer_batch  # version 2.0
    assert InferenceMetadata(multi_core.with_overrides({"version": "3.0"})).richer_batch


def test_shapes(atmos_core):
    metadata = InferenceMetadata(atmos_core)
    n_inputs = len(metadata.dataset_view.data.variable_indices())

    # two input steps, one output step (see the task of the checkpoint)
    assert metadata.input_shape == (1, 2, metadata.number_of_grid_points, n_inputs)
    assert metadata.output_shape[:2] == (1, 1)


@pytest.mark.parametrize("section", ["training", "task"])
def test_multi_dataset_shapes(multi_core, section):
    """Richer-batch checkpoints drop the batch dimension and have an ensemble dimension of 1."""

    def patched(version: str, config: dict) -> InferenceMetadata:
        return InferenceMetadata(multi_core.with_overrides({"version": version, "config": config}))

    reference = InferenceMetadata(multi_core)
    n_inputs = len(reference.dataset_view.data.variable_indices())
    n_outputs = len(reference.dataset_view.data.output_variable_indices())
    n_grid = reference.number_of_grid_points

    # without `multistep_output` in the config (older checkpoints): no time dimension on the output
    metadata = patched("2.0", {"training": {}, "task": {}})
    n_in = metadata.input_shape[1]
    assert metadata.input_shape == (1, n_in, n_grid, n_inputs)
    assert metadata.output_shape == (1, 1, n_grid, n_outputs)
    assert metadata.data_type == "gridded"

    # with it, in `training` or in `task`: (batch, time, ensemble, grid, variables)
    config = {section: {"multistep_output": 3}}
    metadata = patched("2.0", config)
    assert metadata.output_shape == (1, metadata.output_shape[1], 1, n_grid, n_outputs)

    # richer batch: no batch dimension, (time, ensemble, grid, variables)
    metadata = patched("3.0", config)
    assert metadata.richer_batch
    assert metadata.input_shape == (n_in, 1, n_grid, n_inputs)
    assert metadata.output_shape == (metadata.output_shape[0], 1, n_grid, n_outputs)


def test_coordinates(atmos_core, caplog):
    arrays = {"data": {"latitudes": np.arange(5.0), "longitudes": np.arange(5.0) * 2}}

    # without grid_indices the stored coordinates
    metadata = InferenceMetadata(atmos_core.with_overrides({}, arrays))
    latitudes, longitudes = metadata.coordinates()
    assert list(latitudes) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert list(longitudes) == [0.0, 2.0, 4.0, 6.0, 8.0]

    # no arrays: nothing
    assert InferenceMetadata(atmos_core).coordinates() == (None, None)

    # grid_indices keep a subset of the points (deprecated: warned about)
    arrays["data"]["grid_indices"] = np.array([1, 3])
    metadata = InferenceMetadata(atmos_core.with_overrides({}, arrays))
    with caplog.at_level("WARNING"):
        latitudes, longitudes = metadata.coordinates()
    assert list(latitudes) == [1.0, 3.0]
    assert list(longitudes) == [2.0, 6.0]
    assert metadata.number_of_grid_points == 2
    assert any("grid_indices" in record.message for record in caplog.records)


def test_number_of_grid_points(atmos_core):
    metadata = InferenceMetadata(atmos_core)
    assert metadata.number_of_grid_points == atmos_core.dataset().data.grid_size()

    # the length of the arrays wins over the recorded size, an explicit count wins over the arrays
    arrays = {"data": {"latitudes": np.zeros(5), "longitudes": np.zeros(5)}}
    core = atmos_core.with_overrides({}, arrays)
    assert InferenceMetadata(core).number_of_grid_points == 5
    assert InferenceMetadata(core, number_of_grid_points=7).number_of_grid_points == 7
    assert InferenceMetadata(core, number_of_grid_points=7).input_shape[2] == 7


def test_select_variables(atmos_core):
    metadata = InferenceMetadata(atmos_core)

    with_mars = metadata.select_variables(include=["forcing"], has_mars_requests=True)
    variables_metadata = metadata.dataset_view.data.variables_metadata()
    assert with_mars and all("mars" in variables_metadata[v] for v in with_mars)
    assert "cos_latitude" not in with_mars  # computed forcings have no MARS request
    assert "cos_latitude" in metadata.select_variables(include=["forcing"])

    variables, mask = metadata.select_variables_and_masks(include=["prognostic"])
    assert [metadata.dataset_view.data.variable_indices()[v] for v in variables] == list(mask)

    with pytest.raises(ValueError, match="overlap"):
        metadata.select_variables(include=["prognostic"], exclude=["prognostic"])


def test_open_dataset_args_kwargs_removes_unsupported_keys():
    """Test that trajectory key is removed from kwargs but frequency and drop are kept."""
    model_metadata = mock_load_metadata("unit/checkpoints/atmos.json", supporting_arrays=False)

    # Add dataloader config with trajectory key that should be removed
    model_metadata["config"]["dataloader"] = {
        "training": {
            "dataset": "test_dataset",
            "start": 2021,
            "end": 2021,
            "frequency": "6h",
            "drop": [],
            "trajectory": None,
        }
    }

    from anemoi.metadata import Metadata

    metadata = InferenceMetadata(Metadata.from_dict(model_metadata))

    args, kwargs = metadata.open_dataset_args_kwargs(use_original_paths=True, from_dataloader="training")

    # Verify that trajectory is removed
    assert "trajectory" not in kwargs

    # Verify that frequency and drop are kept (they are valid parameters)
    assert kwargs.get("frequency") == "6h"
    assert kwargs.get("drop") == []

    # Verify that other keys are still present
    assert kwargs.get("dataset") == "test_dataset"
    assert kwargs.get("start") == 2021
    assert kwargs.get("end") == 2021


def test_open_dataset_args_kwargs_removes_unsupported_keys_multi_dataset():
    """Test that trajectory key is removed from kwargs but frequency and drop are kept for multi-dataset."""
    model_metadata = mock_load_metadata("unit/checkpoints/multi-single.json", supporting_arrays=False)

    # Add dataloader config with trajectory key that should be removed for multi-dataset structure
    model_metadata["config"]["dataloader"] = {
        "training": {
            "datasets": {
                "data": {
                    "dataset": "test_dataset_multi",
                    "start": 2022,
                    "end": 2022,
                    "frequency": "12h",
                    "drop": ["var1"],
                    "trajectory": None,
                }
            }
        }
    }

    from anemoi.metadata import Metadata

    metadata = InferenceMetadata(Metadata.from_dict(model_metadata), "data")

    args, kwargs = metadata.open_dataset_args_kwargs(use_original_paths=True, from_dataloader="training")

    assert "trajectory" not in kwargs
    assert kwargs.get("frequency") == "12h"
    assert kwargs.get("drop") == ["var1"]
    assert kwargs.get("dataset") == "test_dataset_multi"
    assert kwargs.get("start") == 2022
    assert kwargs.get("end") == 2022


@pytest.mark.parametrize(
    "typed_variables, field_metadata, expected_name",
    [
        # regular case: the GRIB `param` already matches the checkpoint variable name
        pytest.param(
            {"z": z},
            {"param": "z", "levelist": None, "levtype": "sfc"},
            "z",
            id="regular-surface",
        ),
        pytest.param(
            {"w_100": w_100},
            {"param": "w", "levelist": 100, "levtype": "pl"},
            "w_100",
            id="regular-pressure",
        ),
        # renamed case: the GRIB `param` differs from the checkpoint variable name
        pytest.param(
            {"z": z_renamed},
            {"param": "FIS", "levelist": None, "levtype": "sfc"},
            "z",
            id="renamed-surface",
        ),
        pytest.param(
            {"z_100": z_100_renamed},
            {"param": "FI", "levelist": 100, "levtype": "pl"},
            "z_100",
            id="renamed-pressure",
        ),
        # ambiguous case: two different variable names share the same (param, level)
        # key, so the override map must drop that key and fall back to the plain
        # param name instead of arbitrarily picking one of the ambiguous names
        pytest.param(
            {"z": z, "z_alias": z},
            {"param": "z", "levelist": None, "levtype": "sfc"},
            "z",
            id="ambiguous-collision-falls-back-to-param",
        ),
    ],
)
def test_default_namer(typed_variables, field_metadata, expected_name):
    # Bypass the constructor (which expects a real checkpoint) and give the wrapper a dataset view
    # whose `data.typed_variables()` returns the variables of the case.
    metadata = InferenceMetadata.__new__(InferenceMetadata)
    metadata.dataset_view = SimpleNamespace(data=SimpleNamespace(typed_variables=lambda: typed_variables))

    namer = metadata.default_namer()

    assert namer(None, field_metadata) == expected_name
