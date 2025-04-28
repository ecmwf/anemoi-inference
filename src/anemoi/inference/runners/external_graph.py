import logging
from copy import deepcopy
from functools import cached_property
import os
import torch
from ..runners.default import DefaultRunner
from . import runner_registry

LOG = logging.getLogger(__name__)


@runner_registry.register("external_graph")
class ExternalGraphRunner(DefaultRunner):
    """Runner where the graph saved in the checkpoint is replaced by an externally provided one.
      Currently only supported as an extension of the default runner."""
    def __init__(self, config):
        super().__init__(config)

    @cached_property
    def graph(self):
        graph_path = self.config.graph
        assert os.path.isfile(graph_path), f"No graph found at {graph_path}. An external graph needs to be specified in the config file for this runner."
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

        LOG.info(
            "Successfully built model with external graph and reassigning model weights!"
        )
        model_instance.load_state_dict(new_state_dict)
        return model_instance.to(self.device)