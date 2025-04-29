import logging
import os
from copy import deepcopy
from functools import cached_property

import torch

from ..runners.default import DefaultRunner
from . import runner_registry

LOG = logging.getLogger(__name__)


@runner_registry.register("external_graph")
class ExternalGraphRunner(DefaultRunner):
    """Runner where the graph saved in the checkpoint is replaced by an externally provided one.
    Currently only supported as an extension of the default runner.
    """

    def __init__(self, config):
        super().__init__(config)
        # Check if the external graph has the 'indices_connected_nodes' attribute 
        data = self.checkpoint._metadata._config.graph.data
        assert data in self.graph.node_types, f"Node type {data} not found in external graph."
        if 'indices_connected_nodes' in self.graph[data]:
            LOG.info("The external graph has the 'indices_connected_nodes' attribute." \
                    " Patching metadata with MaskedGrid grid_indices.")
            self.checkpoint._metadata.patch(
            {"config": {
                "dataloader" : {
                    "grid_indices" : {
                        "_target_": "anemoi.training.data.grid_indices.MaskedGrid",
                        "nodes_name": data,
                        "node_attribute_name": "indices_connected_nodes",
                    }
                }
            }
            }
            )
            LOG.info("Moving 'grid_indices' from external graph to supporting arrays.")
            indices_connected_nodes = self.graph[data]['indices_connected_nodes'].numpy()
            self.checkpoint._supporting_arrays['grid_indices'] = indices_connected_nodes.squeeze()


    @cached_property
    def graph(self):
        graph_path = self.config.graph
        assert os.path.isfile(
            graph_path
        ), f"No graph found at {graph_path}. An external graph needs to be specified in the config file for this runner."
        LOG.info("Loading external graph from path {graph_path}.")
        return torch.load(graph_path, map_location="cpu", weights_only=False)

    @cached_property
    def model(self):
        model_instance = super().model.to("cpu")
        state_dict = deepcopy(model_instance.state_dict())

        model_instance.graph_data = self.graph
        model_instance.config = self.checkpoint._metadata._config
        
        model_instance._build_model()

        new_state_dict = model_instance.state_dict()

        for key in new_state_dict:
            if key in state_dict and state_dict[key].shape != new_state_dict[key].shape:
                # These are parameters like data_latlon, which are different now because of the graph
                pass
            else:
                # Overwrite with the old parameters
                new_state_dict[key] = state_dict[key]

        LOG.info("Successfully built model with external graph and reassigning model weights!")
        # model_instance.load_state_dict(new_state_dict)
        return model_instance.to(self.device)
