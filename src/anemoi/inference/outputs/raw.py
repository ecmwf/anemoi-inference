# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Raw output: one compressed ``.npz`` file per output step.

Optionally accompanied by ``manifest.json`` and ``grid.npz``, which describe the
checkpoint, variables and grid the files were written with. See
:meth:`RawOutput.write_manifest`.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
from anemoi.utils.provenance import gather_provenance_info
from anemoi.utils.provenance import path_md5
from earthkit.data.utils.dates import to_datetime

from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.provenance import OutputManifest
from anemoi.inference.types import State
from anemoi.inference.utils.templating import render_template

from ..decorators import ensure_dir
from ..decorators import format_dataset_name
from ..decorators import main_argument
from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)


@output_registry.register("raw")
@main_argument("dir")
@format_dataset_name("dir")
@ensure_dir("dir")
class RawOutput(Output):
    """Raw output class."""

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        dir: Path,
        template: str = "{date}.npz",
        strftime: str = "%Y%m%d%H%M%S",
        **kwargs,
    ) -> None:
        """Initialise the RawOutput class.

        Parameters
        ----------
        context : dict
            The context.
        metadata : Metadata
            Metadata corresponding to the dataset this output is handling.
        dir : Path
            The directory to save the raw output.
            If the parent directory does not exist, it will be created.
        template : str, optional
            The template for filenames, by default "{date}.npz".
            Variables available are `date`, `basetime` `step`.
        strftime : str, optional
            The date format string, by default "%Y%m%d%H%M%S".
        output_manifest : bool, optional
            Whether to output a manifest file and grid files (for round-trip inference.) The manifest file is per-dir, and will not be
            overwritten if one already exists.
        """
        super().__init__(context, metadata, **kwargs)
        self.dir = dir
        self.template = template
        self.strftime = strftime
        self.output_manifest = output_manifest

        # Both manifest and grid cover all files in dir/.
        self.manifest_path = Path(self.dir) / "manifest.json"
        self.grid_path = Path(self.dir) / "grid.npz"

    def __repr__(self) -> str:
        """Return a string representation of the RawOutput object.

        Returns
        -------
        str
            String representation of the RawOutput object.
        """
        return f"RawOutput({self.dir})"

    def write_step(self, state: State) -> None:
        """Write the state to a compressed .npz file.

        Parameters
        ----------
        state : State
            The state to be written.
        """
        date = state["date"]
        basetime = date - state["step"]

        if self.output_manifest and ("{basetime" in self.template or "{step" in self.template):
            # The manifest records the reference date, so a reader can only rebuild
            # these filenames if it is the basetime the steps were written against.
            assert basetime == to_datetime(self.reference_date), (
                f"{self}: basetime {basetime} does not match the reference date "
                f"{self.reference_date} recorded in the manifest."
            )

        format_info = {
            "date": date.strftime(self.strftime),
            "step": state["step"],
            "basetime": basetime.strftime(self.strftime),
        }

        fn_state = f"{self.dir}/{render_template(self.template, format_info)}"
        restate = {f"field_{key}": val for key, val in state["fields"].items() if not self.skip_variable(key)}

        for key in ["date"]:
            restate[key] = np.array(state[key], dtype=str)

        # If the lat/lon are not already saved, save them here
        if not self.grid_path.exists():
            for key in ["latitudes", "longitudes"]:
                restate[key] = np.array(state[key])

        np.savez_compressed(fn_state, **restate)

    def open(self, state: State) -> None:
        """Write the sidecar files before the first step, if requested.

        Parameters
        ----------
        state : State
            The initial state.
        """
        if self.output_manifest and not self.manifest_path.exists():
            self.write_manifest(state)

    def write_manifest(self, state: State) -> None:
        """Write the sidecar files describing this set of raw outputs.

        ``manifest.json`` records the checkpoint identity, the variables actually
        written, the filename convention and run-level context. ``grid.npz``
        records the grid once, rather than repeating it in every step file.

        Together they let :class:`RawInput` interpret the ``.npz`` files without
        being told the checkpoint and template out of band.

        Parameters
        ----------
        state : State
            The state the grid is taken from.
        """
        checkpoint_path = str(self.context.checkpoint.path)

        try:
            checkpoint_md5 = path_md5(checkpoint_path)
        except OSError as e:
            LOG.warning("%s: cannot hash checkpoint '%s': %s", self, checkpoint_path, e)
            checkpoint_md5 = None

        manifest = OutputManifest(
            format="anemoi-raw",
            checkpoint={"path": checkpoint_path, "md5": checkpoint_md5},
            dataset_name=self.dataset_name,
            variables=[name for name in self.typed_variables if not self.skip_variable(name)],
            template=self.template,
            strftime=self.strftime,
            grid=self.metadata.grid,
            provenance=gather_provenance_info(),
            reference_date=self.reference_date,
            output_frequency=self._output_frequency,
        )

        with open(self.manifest_path, "w") as f:
            json.dump(asdict(manifest), f, indent=2, default=str)
        LOG.info("%s: wrote %s", self, self.manifest_path)

        latitudes, longitudes = state.get("latitudes"), state.get("longitudes")
        if latitudes is None or longitudes is None:
            LOG.warning("%s: no grid in state, '%s' not written.", self, self.grid_path)
            return

        np.savez_compressed(self.grid_path, latitudes=latitudes, longitudes=longitudes)
