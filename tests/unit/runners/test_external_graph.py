# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runners import create_runner
from anemoi.inference.runners.external_graph import ExternalGraphRunner
from anemoi.inference.testing import fake_checkpoints


class FakeGraph(dict):
    """A graph with the parts the runner reads: node types, node attributes and the ``data`` nodes."""

    @property
    def node_types(self):
        return list(self)


def _graph(**data_attributes) -> FakeGraph:
    return FakeGraph(data=data_attributes)


def _runner(graph: FakeGraph, **kwargs) -> ExternalGraphRunner:
    config = RunConfiguration(
        checkpoint="unit/checkpoints/simple.ckpt",
        date=-1,
        runner="external_graph",
        device="cpu",
        input="dummy",
        lead_time="12h",
        # the test checkpoint has no graph section: name the data nodes of the fake graph
        patch_metadata={"config": {"graph": {"data": "data"}}},
    )
    with patch.object(ExternalGraphRunner, "graph", new=graph):
        return create_runner(config, graph="unused.pt", **kwargs)


def _data(runner: ExternalGraphRunner):
    return runner.checkpoint.multi_dataset_metadata["data"]


@fake_checkpoints
def test_supporting_arrays_are_replaced() -> None:
    graph = _graph(lats=torch.arange(5.0), lons=torch.arange(5.0) * 2)
    runner = _runner(
        graph,
        update_supporting_arrays={"graph": {"latitudes": "lats", "longitudes": "lons"}},
    )

    arrays = runner.checkpoint.metadata.dataset("data").arrays
    np.testing.assert_array_equal(arrays.latitudes, np.arange(5.0))
    np.testing.assert_array_equal(arrays.longitudes, np.arange(5.0) * 2)
    assert _data(runner).number_of_grid_points == 5


@fake_checkpoints
def test_explicit_number_of_grid_points_wins_over_array_lengths() -> None:
    graph = _graph(lats=torch.arange(5.0), lons=torch.arange(5.0))
    runner = _runner(
        graph,
        update_supporting_arrays={"graph": {"latitudes": "lats", "longitudes": "lons"}},
        updated_number_of_grid_points=7,
    )

    assert _data(runner).number_of_grid_points == 7
    assert _data(runner).input_shape[2] == 7
    assert len(runner.checkpoint.metadata.dataset("data").arrays.latitudes) == 5


@fake_checkpoints
def test_explicit_count_survives_later_replacements_and_is_not_shared() -> None:
    graph = _graph(lats=torch.arange(5.0), lons=torch.arange(5.0))
    runner = _runner(graph, updated_number_of_grid_points=7)
    other = _runner(graph)

    # the wrappers rebuilt after a replacement still see the count of the runner
    runner._replace_metadata(supporting_arrays={"latitudes": np.zeros(4), "longitudes": np.zeros(4)})
    assert len(runner.checkpoint.metadata.dataset("data").arrays.latitudes) == 4
    assert _data(runner).number_of_grid_points == 7
    assert _data(runner).output_shape[2] == 7

    # a new explicit count replaces the old one
    runner._replace_metadata(number_of_grid_points=8)
    assert _data(runner).number_of_grid_points == 8

    # another runner has its own count
    assert _data(other).number_of_grid_points != 8


@fake_checkpoints
def test_without_explicit_count_the_arrays_decide() -> None:
    runner = _runner(_graph(lats=torch.arange(5.0), lons=torch.arange(5.0)))
    assert _data(runner).number_of_grid_points == 40320  # recorded in the checkpoint metadata

    runner._replace_metadata(supporting_arrays={"latitudes": np.zeros(3), "longitudes": np.zeros(3)})
    assert _data(runner).number_of_grid_points == 3


@fake_checkpoints
def test_number_of_grid_points_from_a_graph_attribute() -> None:
    graph = _graph(lats=torch.arange(5.0), lons=torch.arange(5.0), other=torch.zeros(9))
    runner = _runner(
        graph,
        update_supporting_arrays={"graph": {"latitudes": "lats", "longitudes": "lons"}},
        updated_number_of_grid_points="other",
    )

    assert _data(runner).number_of_grid_points == 9


@fake_checkpoints
def test_output_mask_and_connected_nodes_are_read_through_the_checkpoint() -> None:
    mask = np.array([True, False, True])
    indices = np.array([[0], [2], [4]])
    graph = _graph(mask=torch.tensor(mask), indices_connected_nodes=torch.tensor(indices))
    runner = _runner(graph, output_mask={"nodes_name": "data", "attribute_name": "mask"})

    arrays = runner.checkpoint.metadata.dataset("data").arrays
    np.testing.assert_array_equal(arrays.output_mask, mask)
    np.testing.assert_array_equal(arrays.grid_indices, [0, 2, 4])
    # the metadata patch of the connected nodes is part of the new metadata
    dataloader = runner.checkpoint.metadata.sections.config["dataloader"]
    assert dataloader["grid_indices"]["node_attribute_name"] == "indices_connected_nodes"


@fake_checkpoints
def test_graph_dataset_replaces_shape_arrays_and_explicit_count() -> None:
    graph_ds = SimpleNamespace(
        shape=(10, 5, 1, 11),
        supporting_arrays=lambda: {"latitudes": np.arange(11.0), "longitudes": np.arange(11.0)},
    )
    with patch("anemoi.datasets.open_dataset", return_value=graph_ds):
        runner = _runner(_graph(), graph_dataset="some-dataset")

    arrays = runner.checkpoint.metadata.dataset("data").arrays
    assert len(arrays.latitudes) == 11
    assert _data(runner).number_of_grid_points == 11
    assert runner.checkpoint.metadata.sections.config["dataloader"]["dataset"] == "some-dataset"


@fake_checkpoints
def test_graph_dataset_count_survives_later_replacements() -> None:
    graph_ds = SimpleNamespace(
        shape=(10, 5, 1, 11),
        supporting_arrays=lambda: {"latitudes": np.arange(11.0), "longitudes": np.arange(11.0)},
    )
    graph = _graph(lats=torch.arange(4.0), lons=torch.arange(4.0))
    with patch("anemoi.datasets.open_dataset", return_value=graph_ds):
        runner = _runner(graph, graph_dataset="some-dataset", update_supporting_arrays={"graph": {"latitudes": "lats"}})

    # the count set from the graph dataset wins over the length of the replaced latitudes
    assert len(runner.checkpoint.metadata.dataset("data").arrays.latitudes) == 4
    assert _data(runner).number_of_grid_points == 11


@fake_checkpoints
def test_missing_graph_key_raises() -> None:
    with pytest.raises(KeyError):
        _runner(_graph(), update_supporting_arrays={"graph": {"latitudes": "nope"}})
