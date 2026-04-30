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
    """Patch a checkpoint file with new dataset metadata from the original training dataset.
    If the checkpoint's metadata is already up to date, this command does nothing. Otherwise, it updates the metadata and supporting arrays in-place.
    """

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        command_parser.add_argument("path", help="Path to the checkpoint.")
        command_parser.add_argument(
            "--sanitise",
            action="store_true",
            help="If there is new metadata, sanitise it before patching.",
        )

    def run(self, args: Namespace) -> None:
        """Run the patch command."""
        from anemoi.utils.checkpoints import replace_metadata

        from anemoi.inference.checkpoint import Checkpoint

        checkpoint = Checkpoint(args.path)
        original_metadata, original_supporting_arrays = checkpoint._raw_metadata
        original_metadata = deepcopy(original_metadata)
        original_supporting_arrays = deepcopy(original_supporting_arrays)

        new_metadata, new_supporting_arrays = checkpoint.update_metadata_from_zarr()

        if diff(original_metadata, new_metadata) or diff(original_supporting_arrays, new_supporting_arrays):
            if args.sanitise:
                LOG.info("Sanitising metadata")
                from anemoi.utils.sanitise import sanitise

                new_metadata = sanitise(new_metadata)

            LOG.info("Patching metadata")
            replace_metadata(args.path, new_metadata, new_supporting_arrays)


command = PatchCmd
