# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
import os
from copy import deepcopy
from functools import cached_property
from typing import Any
from typing import Literal

import torch

from ..decorators import main_argument
from ..runners.default import DefaultRunner
from . import runner_registry

LOG = logging.getLogger(__name__)

# Possibly move the function(s) below to anemoi-models or anemoi-utils since it could be used in transfer learning.


def contains_any(key, specifications):
    contained = False
    for specification in specifications:
        if specification in key:
            contained = True
            break
    return contained


def update_state_dict(
    model, external_state_dict, keywords="", ignore_mismatched_layers=False, ignore_additional_layers=False
):
    """Update the model's stated_dict with entries from an external state_dict. Only entries whose keys contain the specified keywords are considered."""

    LOG.info("Updating model state dictionary.")

    if isinstance(keywords, str):
        keywords = [keywords]

    # select relevant part of external_state_dict
    reduced_state_dict = {k: v for k, v in external_state_dict.items() if contains_any(k, keywords)}
    model_state_dict = model.state_dict()

    # check layers and their shapes
    for key in list(reduced_state_dict):
        if key not in model_state_dict:
            if ignore_additional_layers:
                LOG.info("Skipping injection of %s, which is not in the model.", key)
                del reduced_state_dict[key]
            else:
                raise AssertionError(f"Layer {key} not in model. Consider setting 'ignore_additional_layers = True'.")
        elif reduced_state_dict[key].shape != model_state_dict[key].shape:
            if ignore_mismatched_layers:
                LOG.info("Skipping injection of %s due to shape mismatch.", key)
                LOG.info("Model shape: %s", model_state_dict[key].shape)
                LOG.info("Provided shape: %s", reduced_state_dict[key].shape)
                del reduced_state_dict[key]
            else:
                raise AssertionError(
                    "Mismatch in shape of %s. Consider setting 'ignore_mismatched_layers = True'.", key
                )

    # update
    model.load_state_dict(reduced_state_dict, strict=False)
    return model


def _get_supporting_arrays_from_graph(update_supporting_arrays: dict[str, str], graph: Any) -> dict:
    """Update the supporting arrays from the graph data."""
    updated_supporting_arrays = {}
    for key, value in update_supporting_arrays.items():
        if value in graph["data"]:
            updated_supporting_arrays[key] = graph["data"][value]
            LOG.info("Moving attribute '%s' from external graph to supporting arrays as '%s'.", value, key)
        else:
            error_msg = f"Key '{value}' not found in external graph 'data'. Cannot update supporting array: '{key}'."
            raise KeyError(error_msg)
    return updated_supporting_arrays


def _get_supporting_arrays_from_file(update_supporting_arrays: dict[str, str]) -> dict:
    """Update the supporting arrays from a file."""
    updated_supporting_arrays = {}

    import numpy as np

    for key, value in update_supporting_arrays.items():
        if os.path.isfile(value):
            try:
                updated_supporting_arrays[key] = torch.load(value)
            except Exception as e:
                LOG.warning("Failed to load '%s' as a torch tensor. Attempting to load as numpy array.\n%s", value, e)
                updated_supporting_arrays[key] = np.load(value, allow_pickle=True)

            LOG.info("Moving attribute '%s' from file to supporting arrays as '%s'.", value, key)
        else:
            error_msg = f"File '{value}' not found. Cannot update supporting array: '{key}'."
            raise FileNotFoundError(error_msg)
    return updated_supporting_arrays


def get_updated_supporting_arrays(
    update_supporting_arrays: dict[Literal["graph", "file"], dict[str, str]], graph: Any
) -> dict:
    """Get an update dict for the supporting arrays.

    Allows to update the supporting arrays in the checkpoint metadata from the graph data,
    and from files, which will be loaded using torch.load() or np.load().
    """
    updated_supporting_arrays = {}

    for location in update_supporting_arrays:
        if location == "graph":
            updated_supporting_arrays.update(
                _get_supporting_arrays_from_graph(update_supporting_arrays[location], graph)
            )
        elif location == "file":
            updated_supporting_arrays.update(_get_supporting_arrays_from_file(update_supporting_arrays[location]))
        else:
            raise ValueError(f"Unknown location '{location}' in update_supporting_arrays.")

    return updated_supporting_arrays


@runner_registry.register("external_graph")
@main_argument("graph")
class ExternalGraphRunner(DefaultRunner):
    """Runner where the graph saved in the checkpoint is replaced by an externally provided one.
    Currently only supported as an extension of the default runner.
    """

    def __init__(
        self,
        config: dict,
        graph: str,
        *,
        output_mask: dict | None = {},
        graph_dataset: Any | None = None,
        update_supporting_arrays: dict[Literal["graph", "file"], dict[str, str]] | None = None,
        updated_number_of_grid_points: str | int | None = None,
        check_state_dict: bool | None = True,
    ) -> None:
        """Initialise the ExternalGraphRunner.

        Parameters
        ----------
        config : Configuration
            Configuration for the runner.
        graph : str
            Path to the external graph.
        output_mask : dict | None
            Dictionary specifying the output mask.
        graph_dataset : Any | None
            Argument to open_dataset of anemoi-datasets that recreates the dataset used to build the data nodes of the graph.
        update_supporting_arrays: dict[Literal['graph', 'file'], dict[str, str]] | None
            Dictionary specifying how to update the supporting arrays in the checkpoint metadata.
            Allows both 'graph' and 'file' as keys.
            - 'graph': Will pull from the graph['data'] dictionary, key refers to the name in the supporting arrays,
            and value refers to the name in the graph['data'].
            - 'file': Will pull from the file system, key refers to the name in the supporting arrays,
            and value refers to the path to the file containing the numpy array.
                Can be torch.load() or np.load() compatible.
        updated_number_of_grid_points: str | int | None
            If the number of grid points in the graph is different from that in the checkpoint,
            this can be used to update the number of grid points in the checkpoint metadata.
            If is a str, will be used as a key to the graph['data'] dictionary.
            If is an int, will be used as the number of grid points.
        check_state_dict: bool | None
            Boolean specifying if reconstruction of statedict happens as expeceted.
        """
        super().__init__(config)
        self.check_state_dict = check_state_dict
        self.graph_path = graph

        # If graph was build on other dataset, we need to adapt the dataloader
        if graph_dataset is not None:
            from anemoi.datasets import open_dataset

            graph_ds = open_dataset(graph_dataset)
            LOG.info(
                "The external graph was built using a different anemoi-dataset than that in the checkpoint. "
                "Patching metadata to ensure correct data loading."
            )
            self.checkpoint._metadata.patch(
                {
                    "config": {"dataloader": {"dataset": graph_dataset}},
                    "dataset": {"shape": graph_ds.shape},
                }
            )

            # had to use private attributes because cached properties cause problems
            self.checkpoint._metadata._supporting_arrays.update(graph_ds.supporting_arrays())
            if "grid_indices" in self.checkpoint._metadata._supporting_arrays:
                num_grid_points = len(self.checkpoint._metadata._supporting_arrays["grid_indices"])
            else:
                num_grid_points = graph_ds.shape[-1]
            self.checkpoint._metadata.number_of_grid_points = num_grid_points

        # Check if the external graph has the 'indices_connected_nodes' attribute
        # If so adapt dataloader and add supporting array
        data = self.checkpoint._metadata._config.graph.data
        assert data in self.graph.node_types, f"Node type {data} not found in external graph."
        if "indices_connected_nodes" in self.graph[data]:
            LOG.info(
                "The external graph has the 'indices_connected_nodes' attribute."
                "Patching metadata with MaskedGrid 'grid_indices' to ensure correct data loading."
            )
            self.checkpoint._metadata.patch(
                {
                    "config": {
                        "dataloader": {
                            "grid_indices": {
                                "_target_": "anemoi.training.data.grid_indices.MaskedGrid",
                                "nodes_name": data,
                                "node_attribute_name": "indices_connected_nodes",
                            }
                        }
                    }
                }
            )
            LOG.info("Moving 'indices_connected_nodes' from external graph to supporting arrays as 'grid_indices'.")
            indices_connected_nodes = self.graph[data]["indices_connected_nodes"].numpy()
            self.checkpoint._supporting_arrays["grid_indices"] = indices_connected_nodes.squeeze()

        if output_mask:
            nodes = output_mask["nodes_name"]
            attribute = output_mask["attribute_name"]
            self.checkpoint._supporting_arrays["output_mask"] = self.graph[nodes][attribute].numpy().squeeze()
            LOG.info(
                "Moving attribute '%s' of nodes '%s' from external graph to supporting arrays as 'output_mask'.",
                attribute,
                nodes,
            )

        if update_supporting_arrays is not None:
            self.checkpoint._supporting_arrays.update(
                get_updated_supporting_arrays(update_supporting_arrays, self.graph)
            )

        if updated_number_of_grid_points is not None:
            if isinstance(updated_number_of_grid_points, str):
                updated_number_of_grid_points = len(self.graph["data"][updated_number_of_grid_points])
            self.checkpoint._metadata.number_of_grid_points = updated_number_of_grid_points
            LOG.info(
                "Updated number of grid points in the checkpoint metadata to %s.",
                updated_number_of_grid_points,
            )

    @cached_property
    def graph(self):
        graph_path = self.graph_path
        assert os.path.isfile(
            graph_path
        ), f"No graph found at {graph_path}. An external graph needs to be specified in the config file for this runner."
        LOG.info("Loading external graph from path %s.", graph_path)
        return torch.load(graph_path, map_location="cpu", weights_only=False)

    @cached_property
    def model(self):
        # load the model from the checkpoint
        device = self.device
        self.device = "cpu"
        model_instance = super().model
        state_dict_ckpt = deepcopy(model_instance.state_dict())

        # rebuild the model with the new graph
        model_instance.graph_data = self.graph
        model_instance.config = self.checkpoint._metadata._config
        model_instance._build_model()

        # reinstate the weights, biases and normalizer from the checkpoint
        # reinstating the normalizer is necessary for checkpoints that were created
        # using transfer learning, where the statistics as stored in the checkpoint
        # do not match the statistics used to build the normalizer in the checkpoint.
        model_instance = update_state_dict(
            model_instance, state_dict_ckpt, keywords=["bias", "weight", "processors.normalizer"]
        )

        LOG.info("Successfully built model with external graph and reassigned model weights!")
        self.device = device
        return model_instance.to(self.device)
