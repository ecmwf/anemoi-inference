# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import json
import logging
from copy import deepcopy

from . import Command

LOG = logging.getLogger(__name__)


def diff(original, patched):
    def _diff(a, b, *path):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in set(a) | set(b):
                if key in a and key in b:
                    _diff(a[key], b[key], *path, key)
                else:
                    LOG.info(f"Missing key {'.'.join(list(path) + [key])}")
            return

        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                LOG.info(f"Length mismatch {'.'.join(path)}")
                return
            for i, value in enumerate(a):
                _diff(value, b[i], *path, str(i))
            return

        if a != b:
            LOG.info(f"{path} = {a} -> {b}")

    _diff(original, patched)


class PatchCmd(Command):
    """Patch a checkpoint file."""

    need_logging = True
    _cache = {}

    def add_arguments(self, command_parser):
        command_parser.add_argument("path", help="Path to the checkpoint.")
        command_parser.add_argument("--constants", help="--constants=z,lsm,slor,sdor")

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
            metadata, supporting_arrays = Metadata(metadata).patch_metadata(supporting_arrays, root, args.constants)
            if metadata == previous:
                break
            LOG.info("Metadata patched")

        # Normalize the metadata
        metadata = json.loads(json.dumps(metadata))

        if metadata != original_metadata:
            LOG.info("Patching metadata")
            LOG.info("Diff:")
            diff(original_metadata, metadata)
            LOG.info("----")
            assert "sources" in metadata["dataset"]
            replace_metadata(args.path, metadata, supporting_arrays)

    def _find(self, where, what, matches=None):
        if matches is None:
            matches = []

        if isinstance(where, dict):
            for key, value in where.items():
                if value == what:
                    matches.append(value)
                else:
                    self._find(value, what, matches)

        elif isinstance(where, list):
            for item in where:
                self._find(item, what, matches)

        return matches

    def patch_zarr(self, zarr_attributes, metadata):

        matches = self._find(metadata, zarr_attributes)
        assert len(matches) > 0

        # If we have multiple matches, we will
        # handle them in the next iteration
        zarr_attributes = matches[0]

        uuid = zarr_attributes["uuid"]

        for key in ("constants", "variables_metadata"):

            entry = self._fetch(uuid)
            if key in entry["metadata"]:
                zarr_attributes[key] = entry["metadata"][key]
            else:
                LOG.warning(f"No '{key}' found for dataset '{uuid}'")

        return metadata

    def _fetch(self, uuid):
        from anemoi.registry import Dataset
        from anemoi.registry.entry.dataset import DatasetCatalogueEntryList

        if uuid in self._cache:
            return self._cache[uuid]

        LOG.info(f"Fetching metadata for dataset uuid {uuid}")

        match = None
        for e in DatasetCatalogueEntryList().get(params={"uuid": uuid}):
            if match:
                raise ValueError(f"Multiple entries found for uuid {uuid}")
            match = e

        if match is None:
            raise ValueError(f"No entry found for uuid {uuid}")

        name = match["name"]
        LOG.info(f"Dataset is '{name}'")
        LOG.info(f"https://anemoi.ecmwf.int/datasets/{name}")
        entry = Dataset(name)

        self._cache[uuid] = entry.record
        return self._cache[uuid]

    def _uuid_to_name(self, uuid):
        entry = self._fetch(uuid)
        return entry["name"]


command = PatchCmd
