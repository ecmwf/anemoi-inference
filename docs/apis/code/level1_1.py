import datetime

import numpy as np

from anemoi.inference.runners.simple import SimpleRunner

# Create a runner with the checkpoint file
runner = SimpleRunner("checkpoint.ckpt")

# Select a starting date
date = datetime.datetime(2024, 10, 25)

# Assuming that the initial conditions requires two
# dates, e.g. T0 and T-6

multi_step_input = 2

# Define the grid

latitudes = np.linspace(90, -90, 181)  # 1 degree resolution
longitudes = np.linspace(0, 359, 360)

number_of_points = len(latitudes) * len(longitudes)
latitudes, longitudes = np.meshgrid(latitudes, longitudes)

# Create the initial state

input_state = {
    "date": date,
    "latitudes": latitudes,
    "longitudes": longitudes,
    "fields": {
        "2t": np.random.rand(multi_step_input, number_of_points),
        "msl": np.random.rand(multi_step_input, number_of_points),
        "z_500": np.random.rand(multi_step_input, number_of_points),
        ...: ...,
    },
}

# Run the model

for state in runner.run(input_state=input_state, lead_time=240):
    # This is the date of the new state
    print("New state:", state["date"])

    # This is value of a field for that date
    print("Forecasted 2t:", state["fields"]["2t"])
