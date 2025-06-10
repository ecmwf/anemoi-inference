# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
from copy import deepcopy
from functools import cached_property
from typing import Any

import torch

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


@runner_registry.register("external_graph")
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
        update_supporting_arrays: dict[str, str] | None = None,
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
        update_supporting_arrays: dict[str, str] | None
            Dictionary specifying how to update the supporting arrays in the checkpoint metadata.
            Will pull from the graph['data'] dictionary, key refers to the name in the supporting arrays,
            and value refers to the name in the graph['data'].
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
            self.checkpoint._metadata._supporting_arrays = graph_ds.supporting_arrays()
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
            for key, value in update_supporting_arrays.items():
                if value in self.graph["data"]:
                    self.checkpoint._supporting_arrays[key] = self.graph["data"][value]
                    LOG.info(
                        "Moving attribute '%s' from external graph to supporting arrays as '%s'.",
                        value,
                        key,
                    )
                else:
                    LOG.warning(
                        "Key '%s' not found in external graph 'data'. Skipping update of supporting array '%s'.",
                        value,
                        key,
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
