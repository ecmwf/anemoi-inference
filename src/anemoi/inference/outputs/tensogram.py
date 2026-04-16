# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property
from pathlib import Path

import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State

from ..decorators import main_argument
from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)

# Written into base[i]["name"] for lat/lon objects.  These names are
# deliberately NOT in tensogram-xarray's KNOWN_COORD_NAMES so that lat/lon
# objects share the same flat dimension as all field objects rather than each
# spawning its own dimension.  The canonical names are preserved in the "anemoi"
# namespace for downstream consumers.
_COORD_NAME_MAP = {
    "latitude": "grid_latitude",
    "longitude": "grid_longitude",
}


@output_registry.register("tensogram")
@main_argument("path")
class TensogramOutput(Output):
    """Tensogram output class.

    Writes each forecast step as one tensogram message appended to a .tgm file
    or streamed over a TCP socket.  Each message contains lat/lon coordinate
    objects followed by one object per field (or one stacked object per
    pressure-level parameter when ``stack_pressure_levels=True``).

    Per-object metadata is stored under the ``"anemoi"`` namespace in CBOR.
    Message-level metadata (date, step) is stored in ``_extra_["anemoi"]``.
    Dimension-name hints are stored in ``_extra_["dim_names"]`` so the
    tensogram-xarray backend can resolve meaningful names without the reader
    having to pass ``dim_names`` explicitly.

    Supports local paths and remote URLs (s3://, gs://, az://, ...) via fsspec.
    Each step is encoded and written immediately -- no full-forecast buffering.

    Pressure-level stacking
    -----------------------
    When ``stack_pressure_levels=True``, all fields sharing the same ``param``
    are stacked into a single 2-D object of shape ``(n_grid, n_levels)``,
    sorted by level in ascending order.  The per-object metadata stores
    ``"levels": [500, 850, ...]`` (plural) instead of the scalar ``"level"``
    key used for unstacked fields.

    Without stacking (default), every field is a separate 1-D object and the
    scalar ``"level"`` key is always stored when it is present in the
    checkpoint's GRIB keys.
    """

    def __init__(
        self,
        context: Context,
        path: str,
        encoding: str = "none",
        bits: int | None = None,
        compression: str = "zstd",
        dtype: str = "float32",
        storage_options: dict | None = None,
        stack_pressure_levels: bool = False,
        variables: list[str] | None = None,
        post_processors: list[ProcessorConfig] | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
    ) -> None:
        """Initialise TensogramOutput.

        Parameters
        ----------
        context : Context
            The forecast context.
        path : str
            Destination path:

            * Local file path -- e.g. ``"forecast.tgm"`` or ``"/data/out.tgm"``
            * Remote object-store URL -- e.g. ``"s3://bucket/out.tgm"``,
              ``"gs://..."``, ``"az://..."``
        encoding : str, optional
            Encoding stage: "none" (default) or "simple_packing".
        bits : int | None, optional
            Bits per value for "simple_packing". Required when encoding="simple_packing".
        compression : str, optional
            Compression codec: "none", "zstd" (default), "lz4", "szip", "blosc2".
        dtype : str, optional
            Output dtype for field arrays: "float32" (default) or "float64".
            Coordinate arrays (lat/lon) are always float64.
            When encoding="simple_packing", arrays are promoted to float64 automatically.
        storage_options : dict | None, optional
            Options forwarded to fsspec for remote destinations (credentials,
            region, endpoint overrides, etc.).  Ignored for local files.
        stack_pressure_levels : bool, optional
            When True, pressure-level fields sharing the same param are stacked
            into a single (n_grid, n_levels) object sorted by level ascending.
            Metadata stores ``"levels": [...]`` (plural).  Default False.
        variables : list[str] | None, optional
            Restrict output to this subset of variables. None means all variables.
        post_processors : list[ProcessorConfig] | None, optional
            Post-processors applied to each state before writing.
        output_frequency : int | None, optional
            Write every N steps. None means every step.
        write_initial_state : bool | None, optional
            Whether to write the initial state (step 0).
        """
        super().__init__(
            context,
            variables=variables,
            post_processors=post_processors,
            output_frequency=output_frequency,
            write_initial_state=write_initial_state,
        )
        if encoding == "simple_packing" and bits is None:
            raise ValueError("bits must be set when encoding='simple_packing'")
        self.path = path
        self.encoding = encoding
        self.bits = bits
        self.compression = compression
        self.dtype = dtype
        self.storage_options = storage_options or {}
        self.stack_pressure_levels = stack_pressure_levels
        self._handle = None

    def __repr__(self) -> str:
        return f"TensogramOutput({self.path})"

    @cached_property
    def _numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, state: State) -> None:
        """Open the output stream.

        For local paths, creates parent directories and opens a binary file
        via fsspec.  For remote URLs (``s3://``, ``gs://``, ``az://``, ...),
        opens a writable fsspec stream.

        ``@ensure_path`` is intentionally not used so that ``self.path`` stays
        a ``str``; the fsspec path detection relies on string prefix checks.
        """
        import fsspec

        path_str = str(self.path)
        if "://" not in path_str:
            Path(path_str).parent.mkdir(parents=True, exist_ok=True)

        self._handle = fsspec.open(path_str, "wb", **self.storage_options).open()
        LOG.info("TensogramOutput: writing to %s", path_str)

    def write_initial_state(self, state: State) -> None:
        """Write the initial state, reducing multi-step fields to the last step."""
        from anemoi.inference.state import reduce_state

        state = reduce_state(state)
        return super().write_initial_state(state)

    def write_step(self, state: State) -> None:
        """Encode one forecast step as a tensogram message and write it immediately."""
        if self._handle is None:
            raise RuntimeError(f"{self!r}: write_step called before open() or after close()")

        global_meta = {
            "version": 2,
            "base": [],
            "_extra_": {
                "anemoi": {
                    "date": state["date"].isoformat(),
                    "step": state["step"].total_seconds(),
                }
            },
        }
        descriptors_and_data = []

        # Coordinate objects -- always float64, no lossy encoding.
        for coord_name, coord_arr in [
            ("latitude", state["latitudes"]),
            ("longitude", state["longitudes"]),
        ]:
            arr = np.asarray(coord_arr, dtype=np.float64)
            global_meta["base"].append(
                {
                    "name": _COORD_NAME_MAP[coord_name],
                    "anemoi": {"variable": coord_name},
                }
            )
            descriptors_and_data.append(
                (
                    {"type": "ntensor", "shape": list(arr.shape), "dtype": "float64"},
                    arr,
                )
            )

        # Field objects.
        if self.stack_pressure_levels:
            self._add_fields_stacked(state, global_meta, descriptors_and_data)
        else:
            self._add_fields_flat(state, global_meta, descriptors_and_data)

        # Embed dimension-name hints in _extra_["dim_names"] (generic, no
        # namespace) so the tensogram-xarray backend can replace dim_N fallback
        # names.  Grid axis → "values".  When stacking, also map each unique
        # level-axis size → "level".
        n_grid = len(state["latitudes"])
        dim_names_hint: dict[str, str] = {str(n_grid): "values"}
        if self.stack_pressure_levels:
            for _, arr in descriptors_and_data:
                if arr.ndim == 2:
                    level_size = str(arr.shape[1])
                    if level_size not in dim_names_hint:
                        dim_names_hint[level_size] = "level"
        global_meta["_extra_"]["dim_names"] = dim_names_hint

        import tensogram

        msg_bytes = tensogram.encode(global_meta, descriptors_and_data)
        self._handle.write(msg_bytes)

    def close(self) -> None:
        """Flush and close the output stream."""
        if self._handle is not None:
            try:
                self._handle.flush()
            except Exception:
                pass
            self._handle.close()
            self._handle = None

    # ------------------------------------------------------------------
    # Field object builders
    # ------------------------------------------------------------------

    def _add_fields_flat(
        self,
        state: State,
        global_meta: dict,
        descriptors_and_data: list,
    ) -> None:
        """Add one object per field (default, no stacking)."""
        for name, values in state["fields"].items():
            if self.skip_variable(name):
                continue
            variable = self.typed_variables.get(name)
            if variable is None:
                LOG.warning("TensogramOutput: no typed variable for %r -- metadata will be incomplete", name)
            grib = getattr(variable, "grib_keys", {}) if variable else {}
            base_entry, descriptor, arr = self._build_field_object(name, grib, values)
            global_meta["base"].append(base_entry)
            descriptors_and_data.append((descriptor, arr))

    def _add_fields_stacked(
        self,
        state: State,
        global_meta: dict,
        descriptors_and_data: list,
    ) -> None:
        """Group pressure-level fields by param and stack; write others flat."""
        # pl_groups[param] = [(level, name, grib, values), ...]
        pl_groups: dict[str, list[tuple[int, str, dict, np.ndarray]]] = {}
        non_pl: list[tuple[str, dict, np.ndarray]] = []

        for name, values in state["fields"].items():
            if self.skip_variable(name):
                continue
            variable = self.typed_variables.get(name)
            if variable is None:
                LOG.warning("TensogramOutput: no typed variable for %r -- metadata will be incomplete", name)
            grib = getattr(variable, "grib_keys", {}) if variable else {}
            if variable is not None and variable.is_pressure_level:
                param = variable.param
                level = variable.level
                pl_groups.setdefault(param, []).append((level, name, grib, values))
            else:
                non_pl.append((name, grib, values))

        if not pl_groups:
            LOG.warning("TensogramOutput: stack_pressure_levels=True but no pressure-level fields found")

        # Non-PL fields: one object each.
        for name, grib, values in non_pl:
            base_entry, descriptor, arr = self._build_field_object(name, grib, values)
            global_meta["base"].append(base_entry)
            descriptors_and_data.append((descriptor, arr))

        # PL groups: one stacked object per param, sorted by level ascending.
        for param in sorted(pl_groups):
            group = sorted(pl_groups[param], key=lambda x: x[0])
            base_entry, descriptor, arr = self._build_stacked_object(param, group)
            global_meta["base"].append(base_entry)
            descriptors_and_data.append((descriptor, arr))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_array(self, values: np.ndarray) -> np.ndarray:
        """Cast values to the configured dtype, promoting to float64 for simple_packing."""
        arr = np.asarray(values, dtype=self._numpy_dtype)
        if self.encoding == "simple_packing":
            arr = arr.astype(np.float64)
        return arr

    def _build_descriptor(self, arr: np.ndarray) -> dict:
        """Build the tensogram descriptor dict for a prepared array."""
        descriptor = {
            "type": "ntensor",
            "shape": list(arr.shape),
            "dtype": arr.dtype.name,
            "encoding": self.encoding,
            "compression": self.compression,
        }
        if self.encoding == "simple_packing" and self.bits is not None:
            import tensogram

            # compute_packing_params requires a 1-D float64 array.
            sp_params = tensogram.compute_packing_params(arr.ravel(), self.bits, 0)
            descriptor.update(sp_params)
        return descriptor

    def _build_field_object(
        self,
        name: str,
        grib: dict,
        values: np.ndarray,
    ) -> tuple[dict, dict, np.ndarray]:
        """Build (base_entry, descriptor, array) for a single flat field object."""
        anemoi_meta = {"variable": name, "param": grib.get("param", name)}
        for k in ("levtype", "level"):
            if k in grib:
                anemoi_meta[k] = grib[k]

        # "name" at the top level lets the tensogram-xarray backend resolve
        # the variable name without knowing the "anemoi" namespace.
        base_entry = {"name": name, "anemoi": anemoi_meta}
        arr = self._prepare_array(values)
        return base_entry, self._build_descriptor(arr), arr

    def _build_stacked_object(
        self,
        param: str,
        group: list[tuple[int, str, dict, np.ndarray]],
    ) -> tuple[dict, dict, np.ndarray]:
        """Build (base_entry, descriptor, array) for a stacked pressure-level object.

        Parameters
        ----------
        param : str
            The shared GRIB parameter name for this group.
        group : list
            Sorted list of ``(level, variable_name, grib_keys, values)`` tuples,
            already sorted by level ascending.

        Returns
        -------
        tuple[dict, dict, np.ndarray]
            ``(base_entry, descriptor, array)`` ready to append to the message.
            The array has shape ``(n_grid, n_levels)`` -- grid axis first so that
            all objects in the message share the same leading dimension in the
            tensogram-xarray backend.
        """
        levels = [item[0] for item in group]
        first_grib = group[0][2]

        arrays = [self._prepare_array(item[3]) for item in group]
        # (n_grid, n_levels): grid axis first so all objects share dim "values"
        # in the tensogram-xarray backend.
        stacked = np.column_stack(arrays)

        anemoi_meta = {
            "variable": param,
            "param": param,
            "levtype": first_grib.get("levtype", "pl"),
            "levels": levels,
        }
        # "name" at the top level lets the tensogram-xarray backend resolve
        # the variable name without knowing the "anemoi" namespace.
        base_entry = {"name": param, "anemoi": anemoi_meta}
        return base_entry, self._build_descriptor(stacked), stacked
