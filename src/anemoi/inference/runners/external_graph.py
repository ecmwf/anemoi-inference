import logging
import os
from copy import deepcopy
from functools import cached_property
from typing import Any

import torch
from anemoi.datasets import open_dataset

from ..runners.default import DefaultRunner
from . import runner_registry

LOG = logging.getLogger(__name__)


# Possibly move the function(s) below to anemoi-models or anemoi-utils since it could be used in transfer learning.
def inject_weights_and_biases(model, state_dict, ignore_mismatched_layers=False, ignore_additional_layers=False):
    LOG.info("Updating model weights and biases by injection from an external state dictionary.")
    # select weights and biases from state_dict
    weight_bias_dict = {k: v for k, v in state_dict.items() if "bias" in k or "weight" in k}
    model_state_dict = model.state_dict()

    # check layers and their shapes
    for key in list(weight_bias_dict):
        if key not in model_state_dict:
            if ignore_additional_layers:
                LOG.info(f"Skipping injection of {key}, which is not in the model.")
                del weight_bias_dict[key]
            else:
                raise AssertionError(f"Layer {key} not in model. Consider setting 'ignore_additional_layers = True'.")
        elif weight_bias_dict[key].shape != model_state_dict[key].shape:
            if ignore_mismatched_layers:
                LOG.info(f"Skipping injection of {key} due to shape mismatch.")
                LOG.info(f"Model shape: {model_state_dict[key].shape}")
                LOG.info(f"Provided shape: {weight_bias_dict[key].shape}")
                del weight_bias_dict[key]
            else:
                raise AssertionError(f"Mismatch in shape of {key}. Consider setting 'ignore_mismatched_layers = True'.")

    # inject
    model.load_state_dict(weight_bias_dict, strict=False)
    return model


def contains(key, specifications):
    contained = False
    for specification in specifications:
        if specification in key:
            contained = True
            break
    return contained


def equal_entries(state_dict_1, state_dict_2, layer_specifications):
    equal = True
    keys_1 = [key for key in state_dict_1 if contains(key, layer_specifications)]
    keys_2 = [key for key in state_dict_2 if contains(key, layer_specifications)]
    if not set(keys_1) == set(keys_2):
        equal = False
    else:
        for key in keys_1:
            if not torch.equal(state_dict_1[key], state_dict_2[key]):
                equal = False
                break
    return equal


@runner_registry.register("external_graph")
class ExternalGraphRunner(DefaultRunner):
    """Runner where the graph saved in the checkpoint is replaced by an externally provided one.
    Currently only supported as an extension of the default runner.
    """

    def __init__(
        self,
        config: dict,
        graph: str,
        output_mask: dict | None = {},
        graph_dataset: Any | None = None,
        check_state_dict: bool | None = True,
    ) -> None:
        """Initialize the ExternalGraphRunner.

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
        check_state_dict: bool | None
            Boolean specifying if reconstruction of statedict happens as expeceted.
        """
        super().__init__(config)
        self.check_state_dict = check_state_dict
        self.graph_path = graph

        # If graph was build on other dataset, we need to adapt the dataloader
        if graph_dataset is not None:
            LOG.info(
                "The external graph was built using a different anemoi-dataset than that in the checkpoint."
                "Patching metadata to ensure correct data loading."
            )
            self.checkpoint._metadata.patch(
                {
                    "config": {"dataloader": {"dataset": graph_dataset}},
                    "dataset": {"shape": open_dataset(graph_dataset).shape},
                }
            )
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
                f"Moving attribute '{attribute}' of nodes '{nodes}' from external graph as to supporting arrays 'output_mask'."
            )

    @cached_property
    def graph(self):
        graph_path = self.graph_path
        assert os.path.isfile(
            graph_path
        ), f"No graph found at {graph_path}. An external graph needs to be specified in the config file for this runner."
        LOG.info(f"Loading external graph from path {graph_path}.")
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

        # inject the weights and biases from the checkpoint
        model_instance = inject_weights_and_biases(
            model_instance,
            state_dict_ckpt,
        )

        if self.check_state_dict:
            assert equal_entries(
                model_instance.state_dict(), state_dict_ckpt, ["bias", "weight", "processors.normalizer"]
            ), "Model incorrectly built."

        LOG.info("Successfully built model with external graph and reassiged model weights!")
        self.device = device
        return model_instance.to(self.device)
