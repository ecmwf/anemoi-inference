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

from . import Command

LOG = logging.getLogger(__name__)


class SanitiseCmd(Command):
    """Sanitise a checkpoint file."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : ArgumentParser
            The argument parser to which the arguments will be added.
        """
        command_parser.add_argument("path", help="Path to the checkpoint.")

    def run(self, args: Namespace) -> None:
        """Run the sanitise command.

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """
        from anemoi.utils.checkpoints import load_metadata
        from anemoi.utils.checkpoints import replace_metadata
        from anemoi.utils.sanitise import sanitise

        original_metadata, supporting_arrays = load_metadata(args.path, supporting_arrays=True)
        metadata = deepcopy(original_metadata)
        metadata = sanitise(metadata)

        if metadata != original_metadata:
            LOG.info("Patching metadata")
            assert "sources" in metadata["dataset"]
            replace_metadata(args.path, metadata, supporting_arrays)
        else:
            LOG.info("Metadata is already sanitised")


command = SanitiseCmd
