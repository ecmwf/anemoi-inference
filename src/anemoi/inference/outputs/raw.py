# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Raw output: one ``.npz`` file per output step.

Accompanied by ``manifest.json``, which describes the checkpoint, variables and
filename convention the files were written with (see :meth:`RawOutput.write_manifest`).
While the run is in progress the manifest is named ``manifest.json.part``; it is
renamed to ``manifest.json`` once the run has finished.

Every file is written under a temporary name and renamed into place, so a reader
never sees a partial file.
"""

import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import IO

import numpy as np
from anemoi.utils.provenance import gather_provenance_info
from anemoi.utils.provenance import path_md5
from earthkit.data.utils.dates import to_datetime

from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.provenance import PARTIAL_SUFFIX
from anemoi.inference.provenance import OutputManifest
from anemoi.inference.types import State
from anemoi.inference.utils.templating import render_template

from ..decorators import ensure_dir
from ..decorators import format_dataset_name
from ..decorators import main_argument
from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)


def _write_atomic(path: Path, write: Callable[[IO[bytes]], None]) -> None:
    """Write a file through `write` under a temporary name, then rename it to `path`."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            write(f)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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
        compressed: bool = True,
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
        compressed : bool, optional
            Whether to compress the files, by default True. Compression is slow
            for large states; turn it off when the files are read back soon after.
        """
        super().__init__(context, metadata, **kwargs)
        self.dir = dir
        self.template = template
        self.strftime = strftime
        self.compressed = compressed

        self.manifest_path = Path(self.dir) / "manifest.json"
        self.partial_manifest_path = Path(self.dir) / f"manifest.json{PARTIAL_SUFFIX}"

        if not template.endswith(".npz"):
            raise ValueError(f"{self.__class__.__name__}: template must end with '.npz'.")

    def __repr__(self) -> str:
        """Return a string representation of the RawOutput object.

        Returns
        -------
        str
            String representation of the RawOutput object.
        """
        return f"RawOutput({self.dir})"

    def write_step(self, state: State) -> None:
        """Write the state to a .npz file.

        Parameters
        ----------
        state : State
            The state to be written.
        """
        date = state["date"]
        basetime = date - state["step"]

        if ("{basetime" in self.template or "{step" in self.template) and self.reference_date is not None:
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

        path = Path(self.dir) / render_template(self.template, format_info)
        restate = {f"field_{key}": val for key, val in state["fields"].items() if not self.skip_variable(key)}

        for key in ["date"]:
            restate[key] = np.array(state[key], dtype=str)

        for key in ["latitudes", "longitudes"]:
            restate[key] = np.array(state[key])

        savez = np.savez_compressed if self.compressed else np.savez
        _write_atomic(path, lambda f: savez(f, **restate))

    def open(self, state: State) -> None:
        """Remove the manifest of a previous run and write this run's partial manifest.

        Parameters
        ----------
        state : State
            The initial state.
        """
        self.manifest_path.unlink(missing_ok=True)
        self.write_manifest(state)

    def close(self) -> None:
        """Mark the set of files as complete by giving the manifest its final name."""
        if self.partial_manifest_path.exists():
            os.replace(self.partial_manifest_path, self.manifest_path)

    def write_manifest(self, state: State) -> None:
        """Write the sidecar files describing this set of raw outputs.

        ``manifest.json.part`` records the checkpoint identity, the variables actually
        written, the filename convention and run-level context. :meth:`close` renames
        it to ``manifest.json``.

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

        text = json.dumps(asdict(manifest), indent=2, default=str)
        _write_atomic(self.partial_manifest_path, lambda f: f.write(text.encode()))
        LOG.info("%s: wrote %s", self, self.partial_manifest_path)
