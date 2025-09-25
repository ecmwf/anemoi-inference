# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from argparse import ArgumentParser
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

from . import Command

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np
    from torch_geometric.data import HeteroData


def format_namespace_as_str(namespace: Namespace) -> str:
    """Format an argparse Namespace object as command-line arguments."""
    args = []

    for key, value in vars(namespace).items():
        if key == "command":
            continue
        if value is None:
            continue

        # Convert underscores to hyphens for command line format
        arg_name = f"--{key.replace('_', '-')}"

        if isinstance(value, bool):
            if value:
                args.append(arg_name)
        elif isinstance(value, list):
            args.append(f"{arg_name} {' '.join(map(str, value))}")
        else:
            args.extend([arg_name, str(value)])

    return " ".join(args)


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


class RedefineCmd(Command):
    """Redefine the graph of a checkpoint file."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : ArgumentParser
            The argument parser to which the arguments will be added.
        """
        command_parser.description = "Redefine the graph of a checkpoint file."
        command_parser.add_argument("path", help="Path to the checkpoint.")

        group = command_parser.add_mutually_exclusive_group(required=True)

        group.add_argument("-g", "--graph", type=Path, help="Path to graph file to use")
        group.add_argument("-y", "--graph_config", type=Path, help="Path to graph config to use")
        group.add_argument(
            "-ll",
            "--latlon",
            type=Path,
            help="Path to coordinate npy, should be of shape (N, 2) with latitudes and longitudes.",
        )
        group.add_argument("-c", "--coords", type=str, help="Coordinates, (North West South East Resolution).", nargs=5)

        command_parser.add_argument(
            "-gr",
            "--global_resolution",
            type=str,
            help="Global grid resolution required with --coords, (e.g. n320, o96).",
        )

        command_parser.add_argument("--save-graph", type=str, help="Path to save the updated graph.", default=None)
        command_parser.add_argument("--output", type=str, help="Path to save the updated checkpoint.", default=None)

    def _get_coordinates(self, args: Namespace) -> tuple["np.ndarray", "np.ndarray"]:
        """Get coordinates from command line arguments.

        Either from files or from coords which are extracted from a MARS request.
        """
        import numpy as np

        if args.latlon is not None:
            latlon = np.load(args.latlon)
            return latlon[:, 0], latlon[:, 1]

        elif args.coords is not None:
            import earthkit.data as ekd

            area = [args.coords[0], args.coords[1], args.coords[2], args.coords[3]]

            resolution = str(args.coords[4])
            if resolution.isdigit():
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
        raise ValueError("No valid coordinates found.")

    def _combine_nodes(
        self, latitudes: "np.ndarray", longitudes: "np.ndarray", global_grid: str
    ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
        """Combine lat/lon nodes with global grid if specified.

        Returns lats, lons, local_mask, global_mask
        """
        import numpy as np
        from anemoi.datasets.grids import cutout_mask
        from anemoi.utils.grids import grids

        global_points = grids(global_grid)

        global_removal_mask = cutout_mask(
            latitudes, longitudes, global_points["latitudes"], global_points["longitudes"]
        )
        lats = np.concatenate([latitudes, global_points["latitudes"][global_removal_mask]])
        lons = np.concatenate([longitudes, global_points["longitudes"][global_removal_mask]])
        local_mask = np.array([True] * len(latitudes) + [False] * sum(global_removal_mask), dtype=bool)

        return lats, lons, local_mask, global_removal_mask

    def _make_data_graph(
        self,
        lats: "np.ndarray",
        lons: "np.ndarray",
        local_mask: "np.ndarray",
        global_mask: "np.ndarray",
        *,
        mask_attr_name: str = "cutout",
        attrs,
    ) -> "HeteroData":
        """Make a data graph with the given lat/lon nodes and attributes."""
        import torch
        from anemoi.graphs.nodes import LatLonNodes
        from torch_geometric.data import HeteroData

        graph = LatLonNodes(lats, lons, name="data").update_graph(HeteroData(), attrs_config=attrs)
        graph["data"][mask_attr_name] = torch.from_numpy(local_mask)
        return graph

    def _make_graph_from_coordinates(
        self, args: Namespace, metadata: dict, supporting_arrays: dict
    ) -> tuple[dict, dict, "HeteroData"]:
        """Make a graph from coordinates given in args."""
        import numpy as np

        if args.global_resolution is None:
            raise ValueError("Global resolution must be specified when generating graph from coordinates.")

        local_lats, local_lons = self._get_coordinates(args)
        LOG.info("Coordinates loaded. Number of local nodes: %d", len(local_lats))
        lats, lons, local_mask, global_mask = self._combine_nodes(local_lats, local_lons, args.global_resolution)

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

        data_graph = self._make_data_graph(
            lats, lons, local_mask, global_mask, mask_attr_name=mask_attr_name, attrs=data_graph.get("attrs", None)
        )
        LOG.info("Created data graph with %d nodes.", data_graph.num_nodes)
        graph = creator.update_graph(data_graph)

        supporting_arrays[f"global/{mask_attr_name}"] = global_mask
        supporting_arrays[f"lam_0/{mask_attr_name}"] = np.array([True] * len(local_lats))

        supporting_arrays["latitudes"] = lats
        supporting_arrays["longitudes"] = lons
        supporting_arrays["grid_indices"] = np.ones(global_mask.shape, dtype=np.int64)

        return metadata, supporting_arrays, graph

    def _update_checkpoint(self, model, metadata, graph: "HeteroData"):
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

    def _check_imports(self):
        """Check if required packages are installed."""
        required_packages = ["anemoi.datasets", "anemoi.graphs", "anemoi.models"]
        from importlib.util import find_spec

        for package in required_packages:
            if find_spec(package) is None:
                raise ImportError(f"{package!r} is required for this command.")

    def run(self, args: Namespace) -> None:
        """Run the redefine command.

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """
        self._check_imports()

        import torch
        from anemoi.utils.checkpoints import load_metadata
        from anemoi.utils.checkpoints import save_metadata

        path = Path(args.path)

        metadata, supporting_arrays = load_metadata(str(path), supporting_arrays=True)

        metadata.setdefault("history", [])
        metadata["history"].append(f"anemoi-inference redefine {format_namespace_as_str(args)}")

        if args.graph is not None:
            LOG.info("Loading graph from %s", args.graph)
            graph = torch.load(args.graph)
        else:
            if args.graph_config is not None:
                from anemoi.graphs.create import GraphCreator
                from torch_geometric.data import HeteroData

                graph = GraphCreator(args.graph_config).update_graph(HeteroData())
            else:
                LOG.info("Generating graph from coordinates...")
                metadata, supporting_arrays, graph = self._make_graph_from_coordinates(
                    args, metadata, supporting_arrays
                )

            if args.save_graph is not None:
                torch.save(graph, args.save_graph)
                LOG.info("Saved updated graph to %s", args.save_graph)

        LOG.info("Updating checkpoint...")

        model = torch.load(str(path), weights_only=False, map_location=torch.device("cpu"))
        model = self._update_checkpoint(model, metadata, graph)
        model_path = args.output if args.output is not None else f"{path.stem}_updated{path.suffix}"

        torch.save(model, model_path)

        save_metadata(
            model_path,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )


command = RedefineCmd
