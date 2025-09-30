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
from pathlib import Path

from . import Command

LOG = logging.getLogger(__name__)


def check_redefine_imports():
    """Check if required packages are installed."""
    required_packages = ["anemoi.datasets", "anemoi.graphs", "anemoi.models"]
    from importlib.util import find_spec

    for package in required_packages:
        if find_spec(package) is None:
            raise ImportError(f"{package!r} is required for this command.")


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


class RedefineCmd(Command):
    """Redefine the graph of a checkpoint file."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : ArgumentParser
            The argument parser to which the arguments will be added.
        """
        command_parser.description = "Redefine the graph of a checkpoint file. If using coordinate specifications, assumes the input to the local domain is already regridded."
        command_parser.add_argument("path", help="Path to the checkpoint.")

        group = command_parser.add_mutually_exclusive_group(required=True)

        group.add_argument("-g", "--graph", type=Path, help="Path to graph file to use")
        group.add_argument("-y", "--graph-config", type=Path, help="Path to graph config to use")
        group.add_argument(
            "-ll",
            "--latlon",
            type=Path,
            help="Path to coordinate npy, should be of shape (N, 2) with latitudes and longitudes.",
        )
        group.add_argument("-c", "--coords", type=str, help="Coordinates, (North West South East Resolution).", nargs=5)

        command_parser.add_argument(
            "-gr",
            "--global-resolution",
            type=str,
            help="Global grid resolution required with --coords, (e.g. n320, o96).",
        )

        command_parser.add_argument("--save-graph", type=str, help="Path to save the updated graph.", default=None)
        command_parser.add_argument("--output", type=str, help="Path to save the updated checkpoint.", default=None)

    def run(self, args: Namespace) -> None:
        """Run the redefine command.

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """
        from anemoi.inference.utils.redefine import create_graph_from_config
        from anemoi.inference.utils.redefine import get_coordinates_from_file
        from anemoi.inference.utils.redefine import get_coordinates_from_mars_request
        from anemoi.inference.utils.redefine import load_graph_from_file
        from anemoi.inference.utils.redefine import make_graph_from_coordinates
        from anemoi.inference.utils.redefine import update_checkpoint

        check_redefine_imports()

        import torch
        from anemoi.utils.checkpoints import load_metadata
        from anemoi.utils.checkpoints import save_metadata

        path = Path(args.path)

        # Load checkpoint metadata and supporting arrays
        metadata, supporting_arrays = load_metadata(str(path), supporting_arrays=True)

        # Add command to history
        metadata.setdefault("history", [])
        metadata["history"].append(f"anemoi-inference redefine {format_namespace_as_str(args)}")

        # Create or load the graph
        if args.graph is not None:
            graph = load_graph_from_file(args.graph)
        elif args.graph_config is not None:
            graph = create_graph_from_config(args.graph_config)
        else:
            # Generate graph from coordinates
            LOG.info("Generating graph from coordinates...")

            # Get coordinates based on input type
            if args.latlon is not None:
                local_lats, local_lons = get_coordinates_from_file(args.latlon)
            elif args.coords is not None:
                local_lats, local_lons = get_coordinates_from_mars_request(args.coords)
            else:
                raise ValueError("No valid coordinates found.")

            metadata, supporting_arrays, graph = make_graph_from_coordinates(
                local_lats, local_lons, args.global_resolution, metadata, supporting_arrays
            )

        # Save graph if requested
        if args.save_graph is not None:
            torch.save(graph, args.save_graph)
            LOG.info("Saved updated graph to %s", args.save_graph)

        # Update checkpoint
        LOG.info("Updating checkpoint...")
        model = torch.load(str(path), weights_only=False, map_location=torch.device("cpu"))
        model = update_checkpoint(model, metadata, graph)

        # Save updated checkpoint
        model_path = args.output if args.output is not None else f"{path.stem}_updated{path.suffix}"
        torch.save(model, model_path)

        save_metadata(
            model_path,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        LOG.info("Updated checkpoint saved to %s", model_path)


command = RedefineCmd
