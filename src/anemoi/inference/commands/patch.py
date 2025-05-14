# (C) Copyright 2024 Anemoi contributors.
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

import numpy as np

from . import Command

LOG = logging.getLogger(__name__)


def diff(old, new, *path):

    result = False

    if isinstance(old, dict) and isinstance(new, dict):
        keys1 = set(old.keys()) if isinstance(old, dict) else set()
        keys2 = set(new.keys()) if isinstance(new, dict) else set()

        for key in keys1.union(keys2):
            if key in old and key in new:
                result = diff(old[key], new[key], *path, key) or result
            elif key in old:
                LOG.info(f"{'.'.join(path + (key,))} removed")
                result = True
            elif key in new:
                LOG.info(f"{'.'.join(path + (key,))} added")
                result = True

        return result

    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            LOG.info(f"Difference at {path}: List lengths differ ({len(old)} != {len(new)})")
            result = True
        for i, (item1, item2) in enumerate(zip(old, new)):
            result = diff(item1, item2, *path, str(i)) or result
        return result

    if isinstance(old, np.ndarray) and isinstance(new, np.ndarray):
        if not np.array_equal(old, new):
            LOG.info(f"Difference at {path}: Array values differ")
            result = True
        return result

    if old != new:
        LOG.info(f"{'.'.join(path)}: {old} != {new}")
        result = True

    return result


class PatchCmd(Command):
    """Patch a checkpoint file."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : ArgumentParser
            The argument parser to which the arguments will be added.
        """
        command_parser.add_argument("path", help="Path to the checkpoint.")
        # command_parser.add_argument("--sanitise", action="store_true", help="Sanitise the metadata.")

    def run(self, args: Namespace) -> None:
        """Run the patch command.

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """
        from anemoi.utils.checkpoints import load_metadata
        from anemoi.utils.checkpoints import metadata_root
        from anemoi.utils.checkpoints import replace_metadata

        from anemoi.inference.metadata import Metadata

        root = metadata_root(args.path)

        original_metadata, original_supporting_arrays = load_metadata(args.path, supporting_arrays=True)
        original_metadata = deepcopy(original_metadata)
        original_supporting_arrays = deepcopy(original_supporting_arrays)

        metadata, supporting_arrays = Metadata(original_metadata).patch_metadata(original_supporting_arrays, root)

        if diff(metadata, original_metadata) or diff(original_supporting_arrays, supporting_arrays):
            LOG.info("Patching metadata")
            assert "sources" in metadata["dataset"]
            replace_metadata(args.path, metadata, supporting_arrays)


command = PatchCmd
