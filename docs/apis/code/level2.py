import datetime

from anemoi.inference.inputs.gribfile import GribFileInput
from anemoi.inference.outputs.gribfile import GribFileOutput
from anemoi.inference.runners import DefaultRunner

# Create a runner with the checkpoint file
runner = DefaultRunner("checkpoint.ckpt")

# Select a starting date
date = datetime.datetime(2024, 10, 25)

input = GribFileInput(runner, "input.grib")
output = GribFileOutput(runner, "output.grib")

input_state = input.create_input_state(date)

# Write the initial state to the output file
output.write_initial_state(input_state)

# Run the model and write the output to the file

for state in runner.run(input_state=input_state, lead_time=240):
    output.write_state(state)

# Close the output file
output.close()
