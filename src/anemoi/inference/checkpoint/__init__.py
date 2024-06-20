# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import json
import logging
import os
from functools import cached_property

from anemoi.utils.checkpoints import has_metadata
from anemoi.utils.checkpoints import load_metadata

from .metadata import Metadata

LOG = logging.getLogger(__name__)


class Checkpoint:
    def __init__(self, path):
        self.path = path
        self._metadata = None
        self._operational_config = None

    def __repr__(self):
        return self.path

    def __getattr__(self, name):
        if self._metadata is None:
            try:
                self._metadata = Metadata.from_metadata(load_metadata(self.path))
            except ValueError:
                if has_metadata(self.path):
                    raise
                self._metadata = Metadata.from_metadata(None)

        return getattr(self._metadata, name)

    def _checkpoint_metadata(self, name):
        return load_metadata(self.path, name)

    @cached_property
    def operational_config(self):
        try:
            result = load_metadata(self.path, "operational-config.json")
            LOG.info(f"Using operational configuration from checkpoint {self.path}")
            return result
        except ValueError:
            pass

        # Check for the operational-config.json file in the model directory
        path = os.path.join(os.path.dirname(self.path), "operational-config.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                result = json.load(f)
                LOG.info(f"Using operational configuration from {path}")
                return result

        LOG.warning("No operational configuration found. Using default configuration.")
        return {}
