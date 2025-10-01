# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np
    from torch_geometric.data import HeteroData


def update_state_dict(
    model,
    external_state_dict,
    keywords: list[str] | None = None,
    ignore_mismatched_layers=False,
    ignore_additional_layers=False,
):
    """Update the model's state_dict with entries from an external state_dict. Only entries whose keys contain the specified keywords are considered."""

    LOG.info("Updating model state dictionary.")

    keywords = keywords or []

    # select relevant part of external_state_dict
    reduced_state_dict = {k: v for k, v in external_state_dict.items() if any(kw in k for kw in keywords)}
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
                raise AssertionError(f"Mismatch in shape of {key}. Consider setting 'ignore_mismatched_layers = True'.")

    model.load_state_dict(reduced_state_dict, strict=False)
    return model


def get_coordinates_from_file(latlon_path: Path) -> tuple["np.ndarray", "np.ndarray"]:
    """Get coordinates from a numpy file.

    Parameters
    ----------
    latlon_path : Path
        Path to coordinate npy, should be of shape (N, 2) with latitudes and longitudes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Latitudes and longitudes arrays
    """
    import numpy as np

    latlon = np.load(latlon_path)
    return latlon[:, 0], latlon[:, 1]


COORDINATE = tuple[float, float, float, float, float]


def get_coordinates_from_mars_request(coords: COORDINATE) -> tuple["np.ndarray", "np.ndarray"]:
    """Get coordinates from MARS request parameters.

    Parameters
    ----------
    coords : COORDINATE
        Coordinates (North West South East Resolution)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Latitudes and longitudes arrays
    """
    import earthkit.data as ekd

    area = [coords[0], coords[1], coords[2], coords[3]]

    resolution = str(coords[4])
    if resolution.replace(".", "", 1).isdigit():
        resolution = f"{resolution}/{resolution}"

    ds = ekd.from_source(
        "mars",
        {
            "AREA": area,
            "GRID": f"{resolution}",
            "param": "2t",
            "date": -2,
            "stream": "oper",
            "type": "an",
            "levtype": "sfc",
        },
    )
    return ds[0].grid_points()  # type: ignore


def combine_nodes_with_global_grid(
    latitudes: "np.ndarray", longitudes: "np.ndarray", global_grid: str
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
    """Combine lat/lon nodes with global grid if specified.

    Returns lats, lons, local_mask, global_mask
    """
    import numpy as np
    from anemoi.datasets.grids import cutout_mask
    from anemoi.utils.grids import grids

    global_points = grids(global_grid)

    global_removal_mask = cutout_mask(latitudes, longitudes, global_points["latitudes"], global_points["longitudes"])
    lats = np.concatenate([latitudes, global_points["latitudes"][global_removal_mask]])
    lons = np.concatenate([longitudes, global_points["longitudes"][global_removal_mask]])
    local_mask = np.array([True] * len(latitudes) + [False] * sum(global_removal_mask), dtype=bool)

    return lats, lons, local_mask, global_removal_mask


def make_data_graph(
    lats: "np.ndarray",
    lons: "np.ndarray",
    local_mask: "np.ndarray",
    global_mask: "np.ndarray",
    reference_node_name: str = "data",
    *,
    mask_attr_name: str = "cutout_mask",
    attrs: dict | None = None,
) -> "HeteroData":
    """Make a data graph with the given lat/lon nodes and attributes."""
    import torch
    from anemoi.graphs.nodes import LatLonNodes
    from torch_geometric.data import HeteroData

    graph = LatLonNodes(lats, lons, name=reference_node_name).update_graph(HeteroData(), attrs_config=attrs)  # type: ignore
    graph[reference_node_name][mask_attr_name] = torch.from_numpy(local_mask).unsqueeze(1)
    return graph


def make_graph_from_coordinates(
    local_lats: "np.ndarray", local_lons: "np.ndarray", global_resolution: str, metadata: dict, supporting_arrays: dict
) -> tuple[dict, dict, "HeteroData"]:
    """Make a graph from coordinates.

    Parameters
    ----------
    local_lats : np.ndarray
        Local latitude coordinates
    local_lons : np.ndarray
        Local longitude coordinates
    global_resolution : str
        Global grid resolution (e.g. n320, o96)
    metadata : dict
        Checkpoint metadata
    supporting_arrays : dict
        Supporting arrays from checkpoint

    Returns
    -------
    tuple[dict, dict, HeteroData]
        Updated metadata, supporting arrays, and graph
    """
    import numpy as np

    if global_resolution is None:
        raise ValueError("Global resolution must be specified when generating graph from coordinates.")

    LOG.info("Coordinates loaded. Number of local nodes: %d", len(local_lats))
    lats, lons, local_mask, global_mask = combine_nodes_with_global_grid(local_lats, local_lons, global_resolution)

    graph_config = deepcopy(metadata["config"]["graph"])
    data_graph = graph_config["nodes"].pop("data")

    from anemoi.graphs.create import GraphCreator
    from anemoi.utils.config import DotDict

    creator = GraphCreator(DotDict(graph_config))

    LOG.info("Updating graph...")
    LOG.debug("Using %r", graph_config)

    def nested_get(d, keys, default=None):
        for key in keys:
            d = d.get(key, {})
        return d or default

    mask_attr_name = nested_get(graph_config, ["nodes", "hidden", "node_builder", "mask_attr_name"], "cutout")

    data_graph_attributes = None
    # if mask_attr_name in data_graph.get("attributes", {}):
    #     data_graph_attributes = {mask_attr_name: data_graph["attributes"][mask_attr_name]}

    LOG.info("Found mask attribute name: %r", mask_attr_name)
    # LOG.info("Found data graph attributes: %s", data_graph_attributes)

    data_graph = make_data_graph(
        lats,
        lons,
        local_mask,
        global_mask,
        reference_node_name="data",
        mask_attr_name=mask_attr_name,
        attrs=data_graph_attributes,
    )

    LOG.info("Created data graph with %d nodes.", data_graph.num_nodes)
    graph = creator.clean(creator.update_graph(data_graph))

    supporting_arrays[f"global/{mask_attr_name}"] = global_mask
    supporting_arrays[f"lam_0/{mask_attr_name}"] = np.array([True] * len(local_lats))

    supporting_arrays["latitudes"] = lats
    supporting_arrays["longitudes"] = lons
    supporting_arrays["grid_indices"] = np.ones(local_mask.shape, dtype=np.int64)

    return metadata, supporting_arrays, graph


def update_checkpoint(model, metadata: dict, graph: "HeteroData"):
    """Update checkpoint with new graph and update state dict."""
    from anemoi.utils.config import DotDict

    state_dict_ckpt = deepcopy(model.state_dict())

    # rebuild the model with the new graph
    model.graph_data = graph
    model.config = DotDict(metadata).config
    model._build_model()

    # reinstate the weights, biases and normalizer from the checkpoint
    # reinstating the normalizer is necessary for checkpoints that were created
    # using transfer learning, where the statistics as stored in the checkpoint
    # do not match the statistics used to build the normalizer in the checkpoint.
    model_instance = update_state_dict(model, state_dict_ckpt, keywords=["bias", "weight", "processors.normalizer"])

    return model_instance


def load_graph_from_file(graph_path: Path) -> "HeteroData":
    """Load graph from file.

    Parameters
    ----------
    graph_path : Path
        Path to graph file

    Returns
    -------
    HeteroData
        Loaded graph
    """
    import torch

    LOG.info("Loading graph from %s", graph_path)
    return torch.load(graph_path, weights_only=False, map_location=torch.device("cpu"))


def create_graph_from_config(graph_config_path: Path) -> "HeteroData":
    """Create graph from configuration file.

    Parameters
    ----------
    graph_config_path : Path
        Path to graph configuration file

    Returns
    -------
    HeteroData
        Created graph
    """
    from anemoi.graphs.create import GraphCreator
    from torch_geometric.data import HeteroData

    return GraphCreator(graph_config_path).update_graph(HeteroData())
