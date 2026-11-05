# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
from argparse import ArgumentParser
from argparse import Namespace
from typing import Any

from anemoi.utils.nexus import add_nexus_record_arguments
from anemoi.utils.nexus import nexus_record
from anemoi.utils.nexus import record_attributes
from anemoi.utils.nexus import write_nexus_record

from . import Command

LOG = logging.getLogger(__name__)


def checkpoint_nexus_record(path: str, attributes: dict[str, Any] | None = None) -> dict[str, Any]:
    """The Nexus record of the checkpoint at *path*: its ``uuid`` and, as
    ``metadata``, the metadata embedded in it with ``size`` (the file size in
    bytes) added; then the *attributes* (owner, projects, licenses, ...).
    Nexus names a model by its uuid unless the attributes give a ``name``.

    Parameters
    ----------
    path : str
        The path of the checkpoint file.
    attributes : dict, optional
        The record attributes (see :mod:`anemoi.utils.nexus`).

    Returns
    -------
    dict
        The record, for ``nexus-client create models UUID --file``.
    """
    from anemoi.utils.checkpoints import load_metadata

    metadata = dict(load_metadata(path))
    uuid = metadata.get("uuid")
    if not uuid:
        raise ValueError(f"{path}: the checkpoint metadata has no 'uuid'")
    metadata["size"] = os.path.getsize(path)
    return nexus_record(uuid=uuid, metadata=metadata, attributes=attributes)


class NexusRecordCmd(Command):
    """Print a checkpoint's record for Anemoi Nexus (``nexus-client create models UUID --file``)."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : ArgumentParser
            The command parser.
        """
        command_parser.add_argument("path", metavar="CHECKPOINT", help="Path of the checkpoint file.")
        add_nexus_record_arguments(command_parser)

    def run(self, args: Namespace) -> None:
        """Print or write the record.

        Parameters
        ----------
        args : Namespace
            The command arguments.
        """
        write_nexus_record(checkpoint_nexus_record(args.path, record_attributes(args)), args.output)


command = NexusRecordCmd
