import datetime
import logging
from collections import defaultdict
from pathlib import Path

import earthkit.data as ekd
import earthkit.regrid as ekr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

from ecmwf.opendata import Client as OpendataClient

from anemoi.inference.outputs.printer import print_state
from anemoi.inference.outputs.printer import print_tangent_linear
from anemoi.inference.perturbation import InputPerturbation
from anemoi.inference.runners.tangent_linear import TangentLinearRunner

LOGGER = logging.getLogger(__name__)


GRID_RESOLUTION = "O96"
# PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"]
# PARAM_SFC = ["10u", "10v", "2t", "msl", "lsm", "z", "slor", "sdor"]
# PARAM_SOIL = ["vsw", "sot"]
PARAM_SFC = ["z", "lsm", "slor", "sdor", "skt", "sp", "msl", "tcw"]
PARAM_PL = ["gh", "t", "u", "v", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SOIL_LEVELS = [1, 2]

DATE = OpendataClient().latest()


def load_state(file) -> dict:
    with np.load(file, allow_pickle=False) as data:
        fields = {k: data[k] for k in data.files}
    state = {"date": datetime.datetime(2025, 8, 29, 6, 0), "fields": fields}
    return state


def get_open_data(param, levelist=[]):
    fields = defaultdict(list)
    # Get the data for the current date ONLY (not the previous date)
    for date in [DATE]:  # [DATE - datetime.timedelta(hours=6), DATE]:
        data = ekd.from_source("ecmwf-open-data", date=date, param=param, levelist=levelist)
        for f in data:
            # Open data is between -180 and 180, we need to shift it to 0-360
            assert f.to_numpy().shape == (721, 1440)
            values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
            # Interpolate the data to from 0.25 to grid
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": GRID_RESOLUTION})
            # Add the values to the list
            name = f"{f.metadata('param')}_{f.metadata('levelist')}" if levelist else f.metadata("param")
            fields[name].append(values)

    # Create a single matrix for each parameter
    for param, values in fields.items():
        fields[param] = np.stack(values)

    return fields


def rename_keys(state: dict, mapping: dict) -> dict:
    for old_key, new_key in mapping.items():
        state[new_key] = state.pop(old_key)

    return state


def transform_GH_to_Z(fields: dict, levels: list[str]) -> dict:
    for level in levels:
        fields[f"z_{level}"] = fields.pop(f"gh_{level}") * 9.80665

    return fields


def load_current_state() -> dict:
    fields = {}
    fields.update(get_open_data(param=PARAM_SFC))
    # fields.update(get_open_data(param=PARAM_SOIL,levelist=SOIL_LEVELS))
    fields.update(get_open_data(param=PARAM_PL, levelist=LEVELS))

    # fields = rename_keys(fields, {'sot_1': 'stl1', 'sot_2': 'stl2', 'vsw_1': 'swvl1', 'vsw_2': 'swvl2'})
    fields = transform_GH_to_Z(fields, LEVELS)

    return dict(date=DATE, fields=fields)


def save_state(state, outfile):
    np.savez(outfile, **state["fields"])


def plot_jvp(state: dict, time_step: int, field: str):
    num_times = state["fields"][field].shape[0]
    assert num_times == 1, f"Expected 1 time step in fields for {field}, got {num_times}!"
    LOGGER.warning(f"Plotting jvp for field {field} at {num_times} times ...")
    fig, axs = plt.subplots(num_times, 1, figsize=(12 * num_times, 9), subplot_kw={"projection": ccrs.PlateCarree()})
    if num_times == 1:
        axs = [axs]

    for i in range(num_times):
        axs[i].set_title(f"JVP: {field} (at {(time_step+1)*6}h)")
        vmin, vmax = np.nanmin(state["jvp"][field][i]), np.nanmax(state["jvp"][field][i])
        lim = max(abs(vmin), abs(vmax))
        cmap_kwargs = dict(cmap="bwr", vmin=-lim, vmax=lim, s=20, transform=ccrs.PlateCarree())
        # LOGGER.warning("SHAPES: longitudes %s, latitudes %s, perturbation %s", state["longitudes"].shape, state["latitudes"].shape, state["jvp"][field].shape)
        # LOGGER.warning("jvp min/max: %.5f / %.5f", np.nanmin(state["jvp"][field][i]), np.nanmax(state["jvp"][field][i]))
        sc = axs[i].scatter(state["longitudes"], state["latitudes"], c=state["jvp"][field][i], **cmap_kwargs)
        import cartopy.feature as cfeature

        axs[i].add_feature(cfeature.COASTLINE)
        # set lat/lon boundaries for plot
        axs[i].set_extent([90, 160, 10, 70], crs=ccrs.PlateCarree())
        gl = axs[i].gridlines(draw_labels=True, linewidth=0.5, color='lightgray', linestyle='--')
        gl.bottom_labels = False
    
        cbar = fig.colorbar(
            sc,
            ax=axs[i],
            orientation='horizontal',
            pad=0.1,      # space between plot and colorbar
            fraction=0.05, # relative size
        )
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.set_ticks_position('top')

    # Remove x and y axes
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.savefig(f"./plots/jvp_{field}_step_{time_step:02d}.png")


def main(initial_conditions_file, ckpt: str = {"huggingface": "ecmwf/aifs-single-1.0"}):
    # Load initial conditions
    if initial_conditions_file.exists():
        input_state = load_state(initial_conditions_file)
        LOGGER.info("DATE is wrong")
    else:
        input_state = load_current_state()
        LOGGER.info("State created")
        save_state(input_state, initial_conditions_file)

    # Load model
    runner = TangentLinearRunner(ckpt, device="cuda")

    # The perturbation has physical units (e.g., K for temperature, m for geopotential height, etc.)
    perturbation = InputPerturbation(
        # 10m perturbation in z_500 input inside a 150km radius around the given location
        ckpt, perturbed_variable="z_500", perturbation_location=(40, 120), perturbation_radius_km=300.0, perturbation_magnitude=10.0
    )

    # Compute sensitivities
    for time_step, state in enumerate(runner.run(input_state=input_state, perturbation=perturbation, lead_time="120h")):
        print_state(state)
        print_tangent_linear(state)
        plot_jvp(state, time_step, "t_850")
        plot_jvp(state, time_step, "z_500")


if __name__ == "__main__":
    # I'm using a single-input step model (multistep inputs are not yet supported!)
    main(
        Path("input_state-o96.npz"),
        ckpt="/lus/h2resw01/scratch/syma/aifs/o96/checkpoint/a32fb94c1fdc4d9e8d103a6c1b7c3212/inference-last.ckpt"
    )
