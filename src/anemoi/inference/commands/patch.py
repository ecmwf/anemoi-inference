# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from copy import deepcopy

from . import Command

LOG = logging.getLogger(__name__)


class PatchCmd(Command):
    """Patch a checkpoint file."""

    need_logging = True
    _cache = {}

    def add_arguments(self, command_parser):
        command_parser.add_argument("path", help="Path to the checkpoint.")
        command_parser.add_argument("--sanitise", action="store_true", help="Sanitise the metadata.")
        command_parser.add_argument("--force", action="store_true", help="Force the patching.")

    def run(self, args):
        from anemoi.utils.checkpoints import load_metadata
        from anemoi.utils.checkpoints import metadata_root
        from anemoi.utils.checkpoints import replace_metadata

        from anemoi.inference.metadata import Metadata

        root = metadata_root(args.path)

        original_metadata, supporting_arrays = load_metadata(args.path, supporting_arrays=True)
        metadata = deepcopy(original_metadata)

        # Patch the metadata
        while True:
            previous = deepcopy(metadata)
            metadata, supporting_arrays = Metadata(metadata).patch_metadata(supporting_arrays, root, force=args.force)
            if metadata == previous:
                break
            LOG.info("Metadata patched")

        if metadata != original_metadata:
            LOG.info("Patching metadata")
            assert "sources" in metadata["dataset"]
            replace_metadata(args.path, metadata, supporting_arrays)


command = PatchCmd
