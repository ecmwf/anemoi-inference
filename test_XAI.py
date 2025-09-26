import datetime
import logging
from collections import defaultdict
from pathlib import Path

import earthkit.data as ekd
import earthkit.regrid as ekr
import matplotlib.pyplot as plt
import numpy as np
from ecmwf.opendata import Client as OpendataClient

from anemoi.inference.outputs.printer import print_state
from anemoi.inference.runners.sensitivities import Perturbation
from anemoi.inference.runners.sensitivities import SensitivitiesRunner

LOGGER = logging.getLogger(__name__)


GRID_RESOLUTION = "O96"
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"]
PARAM_SOIL = ["vsw", "sot"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
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
    # Get the data for the current date and the previous date
    for date in [DATE - datetime.timedelta(hours=6), DATE]:
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


def plot_sensitivities(state: dict, field: str):
    num_times = state["fields"][field].shape[0]
    fig, axs = plt.subplots(num_times, 1, figsize=(6 * num_times, 8))

    # Get the combined min/max for color normalization
    vmin = min(state["fields"][field][0].min(), state["fields"][field][1].min())
    vmax = max(state["fields"][field][0].max(), state["fields"][field][1].max())
    lim = max(abs(vmin), abs(vmax))
    cmap_kwargs = dict(cmap="PuOr", vmin=-lim, vmax=lim)

    for i in range(num_times):
        axs[i].set_title(f"{field} (at -{(num_times-i)*6}H)")
        axs[i].scatter(state["longitudes"], state["latitudes"], c=state["fields"][field][i], **cmap_kwargs)

    # Remove x and y axes
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.savefig(f"sensitivities_{field}.png")


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
    runner = SensitivitiesRunner(ckpt, device="cuda", perturb_normalised_space=True)

    perturbation = Perturbation(
        ckpt, perturbed_variable="2t", perturbation_location=(40, 120), perturbation_radius_km=150.0
    )

    # Compute sensitivities
    for state in runner.run(input_state=input_state, perturbation=perturbation, lead_time="6h"):
        print_state(state)
        plot_sensitivities(state, "2t")
        plot_sensitivities(state, "z")


if __name__ == "__main__":
    main(Path("input_state-o96.npz"), ckpt="../inference-aifs-o96.ckpt")
