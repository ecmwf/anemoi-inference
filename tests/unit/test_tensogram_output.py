# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for TensogramOutput."""

from datetime import datetime
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from anemoi.inference.outputs.tensogram import TensogramOutput

tensogram = pytest.importorskip("tensogram", reason="tensogram not installed")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_GRID = 100
FIELD_NAMES = ["2t", "10u", "10v"]


def _make_variable(param=None, levtype=None, level=None):
    grib = {"param": param} if param is not None else {}
    if levtype is not None:
        grib["levtype"] = levtype
    if level is not None:
        grib["level"] = level
    is_pl = levtype is not None and level is not None
    return SimpleNamespace(
        grib_keys=grib,
        is_computed_forcing=False,
        is_pressure_level=is_pl,
        param=param,
        level=level,
    )


def _make_context(field_names=FIELD_NAMES):
    typed_variables = {name: _make_variable(param=name) for name in field_names}

    checkpoint = SimpleNamespace(
        typed_variables=typed_variables,
    )
    context = SimpleNamespace(
        checkpoint=checkpoint,
        reference_date=datetime(2024, 1, 1),
        write_initial_state=False,
        output_frequency=None,
        typed_variables={},
    )
    return context


def _make_state(step_hours=1, field_names=FIELD_NAMES, n_grid=N_GRID, seed=0):
    rng = np.random.default_rng(seed)
    date = datetime(2024, 1, 1) + timedelta(hours=step_hours)
    return {
        "date": date,
        "step": timedelta(hours=step_hours),
        "latitudes": np.linspace(-90, 90, n_grid, dtype=np.float64),
        "longitudes": np.linspace(0, 360, n_grid, dtype=np.float64),
        "fields": {name: rng.random(n_grid).astype(np.float32) for name in field_names},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_write_step_before_open_raises():
    """write_step raises RuntimeError when called before open()."""
    context = _make_context()
    output = TensogramOutput(context, "/tmp/never.tgm")
    with pytest.raises(RuntimeError, match="open"):
        output.write_step(_make_state())


def test_write_initial_state_reduces_multistep(tmp_path):
    """write_initial_state reduces a multi-step field array to its last step."""
    from unittest.mock import patch

    path = tmp_path / "init.tgm"
    # write_initial_state=True so the base class actually calls write_step.
    context = _make_context(["2t"])
    output = TensogramOutput(context, str(path), write_initial_state=True)

    # Build a state where "2t" has shape (3, N_GRID) -- 3 input steps.
    rng = np.random.default_rng(1)
    multi_step_values = rng.random((3, N_GRID)).astype(np.float32)
    state = {
        "date": datetime(2024, 1, 1),
        "step": timedelta(0),
        "latitudes": np.linspace(-90, 90, N_GRID, dtype=np.float64),
        "longitudes": np.linspace(0, 360, N_GRID, dtype=np.float64),
        "fields": {"2t": multi_step_values},
    }

    written_states = []

    def capture_write_step(s):
        written_states.append(s)

    output.open(state)
    with patch.object(output, "write_step", side_effect=capture_write_step):
        output.write_initial_state(state)
    output.close()

    assert written_states, "write_step was not called by write_initial_state"
    written_field = written_states[0]["fields"]["2t"]
    # reduce_state selects the last step along axis 0.
    np.testing.assert_array_equal(written_field, multi_step_values[-1])


def test_write_and_read_local(tmp_path):
    """Write 3 steps to a local .tgm file and verify round-trip."""
    path = tmp_path / "forecast.tgm"
    context = _make_context()
    output = TensogramOutput(context, str(path))

    states = [_make_state(h) for h in range(1, 4)]
    output.open(states[0])
    for state in states:
        output.write_step(state)
    output.close()

    assert path.exists()

    tgm_file = tensogram.TensogramFile.open(str(path))
    assert len(tgm_file) == 3

    for i, msg in enumerate(tgm_file):
        meta, objects = msg

        # lat + lon + 3 fields = 5 objects per message
        assert len(objects) == 5

        # Coordinate objects -- "name" uses "grid_*" to avoid KNOWN_COORD_NAMES
        # so all objects share one flat dimension in the xarray backend.
        assert meta.base[0]["name"] == "grid_latitude"
        assert meta.base[1]["name"] == "grid_longitude"
        assert meta.base[0]["anemoi"]["variable"] == "latitude"
        assert meta.base[1]["anemoi"]["variable"] == "longitude"
        lat_desc, lat_arr = objects[0]
        lon_desc, lon_arr = objects[1]
        assert lat_desc.dtype == "float64"
        assert lon_desc.dtype == "float64"
        np.testing.assert_allclose(lat_arr, states[i]["latitudes"])
        np.testing.assert_allclose(lon_arr, states[i]["longitudes"])

        # Field objects -- "name" top-level for xarray backend.
        expected_step = int(states[i]["step"].total_seconds() / 3600)
        expected_base_dt = (states[i]["date"] - states[i]["step"]).isoformat()
        for j, name in enumerate(FIELD_NAMES):
            obj_idx = 2 + j
            assert meta.base[obj_idx]["name"] == name
            assert meta.base[obj_idx]["anemoi"]["variable"] == name
            assert meta.base[obj_idx]["mars"]["param"] == name
            assert meta.base[obj_idx]["mars"]["step"] == expected_step
            assert meta.base[obj_idx]["mars"]["basedatetime"] == expected_base_dt
            _, field_arr = objects[obj_idx]
            np.testing.assert_allclose(field_arr, states[i]["fields"][name], rtol=1e-6)

        # Extra metadata
        extra = meta.extra["anemoi"]
        assert extra["step"] == states[i]["step"].total_seconds()
        assert extra["date"] == states[i]["date"].isoformat()


def test_variable_filter(tmp_path):
    """Only the selected variable and coordinates are written."""
    path = tmp_path / "filtered.tgm"
    context = _make_context()
    output = TensogramOutput(context, str(path), variables=["2t"])

    state = _make_state()
    output.open(state)
    output.write_step(state)
    output.close()

    tgm_file = tensogram.TensogramFile.open(str(path))
    msg = tgm_file[0]
    meta, objects = msg

    # lat + lon + 1 filtered field
    assert len(objects) == 3
    assert meta.base[2]["anemoi"]["variable"] == "2t"


def test_simple_packing_encoding(tmp_path):
    """simple_packing round-trip stays within expected quantisation error."""
    path = tmp_path / "packed.tgm"
    context = _make_context(["2t"])
    output = TensogramOutput(context, str(path), encoding="simple_packing", bits=16, compression="zstd")

    rng = np.random.default_rng(42)
    values = (rng.random(N_GRID) * 50 + 250).astype(np.float32)  # ~250-300 K range
    state = {
        "date": datetime(2024, 1, 1, 1),
        "step": timedelta(hours=1),
        "latitudes": np.linspace(-90, 90, N_GRID, dtype=np.float64),
        "longitudes": np.linspace(0, 360, N_GRID, dtype=np.float64),
        "fields": {"2t": values},
    }

    output.open(state)
    output.write_step(state)
    output.close()

    tgm_file = tensogram.TensogramFile.open(str(path))
    meta, objects = tgm_file[0]
    _, decoded = objects[2]  # index 0=lat, 1=lon, 2=2t

    # 16-bit packing over a 50 K range: max error ≈ 50/65535 ≈ 0.001 K
    np.testing.assert_allclose(decoded, values, atol=0.002)


def test_simple_packing_requires_bits(tmp_path):
    """encoding='simple_packing' without bits raises ValueError immediately."""
    context = _make_context(["2t"])
    with pytest.raises(ValueError, match="bits must be set"):
        TensogramOutput(context, str(tmp_path / "out.tgm"), encoding="simple_packing")


def test_mars_metadata_forwarded(tmp_path):
    """grib_keys appear in per-object 'mars' namespace, following tensogram-grib convention.
    step and basedatetime are also written into mars.
    """
    path = tmp_path / "levs.tgm"
    typed_variables = {
        "t500": _make_variable(param="t", levtype="pl", level=500),
    }
    checkpoint = SimpleNamespace(typed_variables=typed_variables)
    context = SimpleNamespace(
        checkpoint=checkpoint,
        reference_date=datetime(2024, 1, 1),
        write_initial_state=False,
        output_frequency=None,
        typed_variables={},
    )

    output = TensogramOutput(context, str(path))
    state = {
        "date": datetime(2024, 1, 1, 6),
        "step": timedelta(hours=6),
        "latitudes": np.zeros(10, dtype=np.float64),
        "longitudes": np.zeros(10, dtype=np.float64),
        "fields": {"t500": np.ones(10, dtype=np.float32)},
    }
    output.open(state)
    output.write_step(state)
    output.close()

    tgm_file = tensogram.TensogramFile.open(str(path))
    meta, _ = tgm_file[0]
    mars = meta.base[2]["mars"]
    assert mars["param"] == "t"
    assert mars["levtype"] == "pl"
    assert mars["level"] == 500
    assert mars["step"] == 6
    assert mars["basedatetime"] == datetime(2024, 1, 1, 0).isoformat()
    # "anemoi" only carries the internal variable name
    assert meta.base[2]["anemoi"]["variable"] == "t500"


def test_remote_write_via_memory_fs():
    """Write to fsspec memory filesystem and read back valid tensogram messages."""
    import fsspec

    url = "memory://test_forecast.tgm"
    context = _make_context()
    output = TensogramOutput(context, url)

    states = [_make_state(h) for h in range(1, 3)]
    output.open(states[0])
    for state in states:
        output.write_step(state)
    output.close()

    # Read back raw bytes via fsspec memory fs and scan as tensogram messages.
    fs = fsspec.filesystem("memory")
    raw = fs.open(url, "rb").read()

    messages = tensogram.scan(raw)
    assert len(messages) == 2

    for i, (offset, length) in enumerate(messages):
        msg_bytes = raw[offset : offset + length]
        meta, objects = tensogram.decode(msg_bytes)
        assert len(objects) == 5  # lat, lon, 3 fields
        assert meta.extra["anemoi"]["step"] == states[i]["step"].total_seconds()


def test_dtype_float64(tmp_path):
    """dtype=float64 stores field arrays as float64."""
    path = tmp_path / "f64.tgm"
    context = _make_context(["2t"])
    output = TensogramOutput(context, str(path), dtype="float64")

    state = _make_state(field_names=["2t"])
    output.open(state)
    output.write_step(state)
    output.close()

    tgm_file = tensogram.TensogramFile.open(str(path))
    meta, objects = tgm_file[0]
    desc, arr = objects[2]
    assert desc.dtype == "float64"
    assert arr.dtype == np.float64


def test_close_is_idempotent(tmp_path):
    """close() can be called multiple times without error."""
    path = tmp_path / "idem.tgm"
    context = _make_context()
    output = TensogramOutput(context, str(path))
    output.open(_make_state())
    output.write_step(_make_state())
    output.close()
    output.close()  # should not raise


def test_dim_names_in_metadata(tmp_path):
    """dim_names hint is written into _extra_[dim_names] and readable by xarray."""
    xr = pytest.importorskip("xarray", reason="xarray not installed")
    pytest.importorskip("tensogram_xarray", reason="tensogram_xarray not installed")

    path = tmp_path / "dimnames.tgm"
    context = _make_context()
    output = TensogramOutput(context, str(path))
    state = _make_state()
    output.open(state)
    output.write_step(state)
    output.close()

    tgm_file = tensogram.TensogramFile.open(str(path))
    meta, _ = tgm_file[0]
    dim_names = meta.extra["dim_names"]
    assert str(N_GRID) in dim_names
    assert dim_names[str(N_GRID)] == "values"

    ds = xr.open_dataset(str(path), engine="tensogram")
    assert "values" in ds.dims
    assert ds["2t"].dims == ("values",)


def test_stacked_dim_names_in_xarray(tmp_path):
    """Stacked fields get (values, level) dims when opened in xarray."""
    xr = pytest.importorskip("xarray", reason="xarray not installed")
    pytest.importorskip("tensogram_xarray", reason="tensogram_xarray not installed")

    path = tmp_path / "stacked_dims.tgm"
    context = _make_pl_context(params=["t"], levels=[500, 850, 1000])
    output = TensogramOutput(context, str(path), stack_pressure_levels=True)
    state = _make_pl_state(params=["t"], levels=[500, 850, 1000])
    output.open(state)
    output.write_step(state)
    output.close()

    ds = xr.open_dataset(str(path), engine="tensogram")
    assert "values" in ds.dims
    assert "level" in ds.dims
    assert ds["t"].dims == ("values", "level")
    assert ds["t"].shape == (N_GRID, 3)


# ---------------------------------------------------------------------------
# Pressure-level stacking tests
# ---------------------------------------------------------------------------

PL_LEVELS = [500, 850, 1000]
PL_PARAMS = ["t", "u"]


def _make_pl_context(params=PL_PARAMS, levels=PL_LEVELS, extra_fields=None):
    """Context with pressure-level variables plus optional non-PL extras."""
    typed_variables = {}
    for param in params:
        for level in levels:
            name = f"{param}{level}"
            typed_variables[name] = _make_variable(param=param, levtype="pl", level=level)
    for name in extra_fields or []:
        typed_variables[name] = _make_variable(param=name)

    checkpoint = SimpleNamespace(typed_variables=typed_variables)
    return SimpleNamespace(
        checkpoint=checkpoint,
        reference_date=datetime(2024, 1, 1),
        write_initial_state=False,
        output_frequency=None,
        typed_variables={},
    )


def _make_pl_state(params=PL_PARAMS, levels=PL_LEVELS, extra_fields=None, n_grid=N_GRID, seed=0):
    rng = np.random.default_rng(seed)
    fields = {}
    for param in params:
        for level in levels:
            fields[f"{param}{level}"] = rng.random(n_grid).astype(np.float32)
    for name in extra_fields or []:
        fields[name] = rng.random(n_grid).astype(np.float32)
    return {
        "date": datetime(2024, 1, 1, 1),
        "step": timedelta(hours=1),
        "latitudes": np.linspace(-90, 90, n_grid, dtype=np.float64),
        "longitudes": np.linspace(0, 360, n_grid, dtype=np.float64),
        "fields": fields,
    }


def test_stack_object_count(tmp_path):
    """With stacking: one object per param, not per level."""
    path = tmp_path / "stacked.tgm"
    context = _make_pl_context()
    output = TensogramOutput(context, str(path), stack_pressure_levels=True)

    state = _make_pl_state()
    output.open(state)
    output.write_step(state)
    output.close()

    tgm_file = tensogram.TensogramFile.open(str(path))
    meta, objects = tgm_file[0]

    # lat + lon + 1 object per param (t, u)
    assert len(objects) == 2 + len(PL_PARAMS)


def test_stack_shape(tmp_path):
    """Stacked objects have shape (n_grid, n_levels) -- grid axis first."""
    path = tmp_path / "stacked.tgm"
    context = _make_pl_context()
    output = TensogramOutput(context, str(path), stack_pressure_levels=True)

    state = _make_pl_state()
    output.open(state)
    output.write_step(state)
    output.close()

    tgm_file = tensogram.TensogramFile.open(str(path))
    meta, objects = tgm_file[0]

    for obj_idx in range(2, 2 + len(PL_PARAMS)):
        desc, arr = objects[obj_idx]
        assert arr.shape == (N_GRID, len(PL_LEVELS)), f"expected ({N_GRID}, {len(PL_LEVELS)}), got {arr.shape}"


def test_stack_levels_metadata(tmp_path):
    """Stacked objects store 'levels' (plural) sorted ascending; no scalar 'level'."""
    path = tmp_path / "stacked.tgm"
    context = _make_pl_context()
    output = TensogramOutput(context, str(path), stack_pressure_levels=True)

    state = _make_pl_state()
    output.open(state)
    output.write_step(state)
    output.close()

    tgm_file = tensogram.TensogramFile.open(str(path))
    meta, _ = tgm_file[0]

    for obj_idx in range(2, 2 + len(PL_PARAMS)):
        entry = meta.base[obj_idx]
        anemoi = entry["anemoi"]
        mars = entry["mars"]
        assert "levels" in anemoi, "stacked objects must have 'levels' key"
        assert "level" not in anemoi, "stacked objects must not have scalar 'level' in anemoi"
        assert "level" not in mars, "stacked objects must not have scalar 'level' in mars"
        assert anemoi["levels"] == sorted(PL_LEVELS)
        assert mars["levtype"] == "pl"
        # "name" top-level for xarray backend matches the param in mars
        assert "name" in entry
        assert entry["name"] == mars["param"]


def test_stack_values_round_trip(tmp_path):
    """Stacked values decode to the same data as the input, in level-sorted order."""
    path = tmp_path / "stacked.tgm"
    context = _make_pl_context(params=["t"], levels=[850, 500, 1000])  # unsorted input
    output = TensogramOutput(context, str(path), stack_pressure_levels=True)

    rng = np.random.default_rng(7)
    fields = {f"t{lv}": rng.random(N_GRID).astype(np.float32) for lv in [850, 500, 1000]}
    state = {
        "date": datetime(2024, 1, 1, 1),
        "step": timedelta(hours=1),
        "latitudes": np.linspace(-90, 90, N_GRID, dtype=np.float64),
        "longitudes": np.linspace(0, 360, N_GRID, dtype=np.float64),
        "fields": fields,
    }

    output.open(state)
    output.write_step(state)
    output.close()

    tgm_file = tensogram.TensogramFile.open(str(path))
    meta, objects = tgm_file[0]

    anemoi = meta.base[2]["anemoi"]
    assert anemoi["levels"] == [500, 850, 1000]  # sorted ascending

    _, arr = objects[2]
    # shape is (n_grid, n_levels); columns are levels sorted ascending [500, 850, 1000]
    np.testing.assert_allclose(arr[:, 0], fields["t500"], rtol=1e-6)
    np.testing.assert_allclose(arr[:, 1], fields["t850"], rtol=1e-6)
    np.testing.assert_allclose(arr[:, 2], fields["t1000"], rtol=1e-6)


def test_stack_non_pl_fields_written_flat(tmp_path):
    """Non-PL fields are written as individual objects even with stacking enabled."""
    path = tmp_path / "mixed.tgm"
    context = _make_pl_context(params=["t"], levels=[500, 850], extra_fields=["2t"])
    output = TensogramOutput(context, str(path), stack_pressure_levels=True)

    state = _make_pl_state(params=["t"], levels=[500, 850], extra_fields=["2t"])
    output.open(state)
    output.write_step(state)
    output.close()

    tgm_file = tensogram.TensogramFile.open(str(path))
    meta, objects = tgm_file[0]

    # lat + lon + 1 stacked t + 1 flat 2t = 4
    assert len(objects) == 4

    # Verify 2t is a 1-D flat field with scalar level metadata (no "levels" key)
    flat_entries = [
        meta.base[i]["anemoi"] for i in range(2, len(objects)) if meta.base[i]["anemoi"].get("variable") == "2t"
    ]
    assert len(flat_entries) == 1
    assert "levels" not in flat_entries[0]
    assert "level" not in flat_entries[0]


def test_no_stack_level_metadata_correct(tmp_path):
    """Without stacking, each PL field stores scalar 'level' and 'levtype'."""
    path = tmp_path / "flat_pl.tgm"
    context = _make_pl_context(params=["t"], levels=[500, 850])
    output = TensogramOutput(context, str(path), stack_pressure_levels=False)

    state = _make_pl_state(params=["t"], levels=[500, 850])
    output.open(state)
    output.write_step(state)
    output.close()

    tgm_file = tensogram.TensogramFile.open(str(path))
    meta, objects = tgm_file[0]

    # lat + lon + 2 individual fields
    assert len(objects) == 4

    for obj_idx in range(2, 4):
        mars = meta.base[obj_idx]["mars"]
        anemoi = meta.base[obj_idx]["anemoi"]
        assert "level" in mars
        assert "levtype" in mars
        assert mars["levtype"] == "pl"
        assert mars["param"] == "t"
        assert "levels" not in anemoi  # no plural key in flat mode
