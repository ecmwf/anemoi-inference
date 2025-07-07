import datetime

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.inputs.gribfile import GribFileInput
from anemoi.inference.outputs.gribout import GribOutput
from anemoi.inference.runners.default import DefaultRunner

# Create a runner with the checkpoint file
runner = DefaultRunner(RunConfiguration(checkpoint="checkpoint.ckpt"))

# Select a starting date
date = datetime.datetime(2024, 10, 25)

input = GribFileInput(runner, "input.grib")
output = GribOutput(runner, "output.grib")

input_state = input.create_input_state(date=date)

# Write the initial state to the output file
output.write_initial_state(input_state)

# Run the model and write the output to the file

for state in runner.run(input_state=input_state, lead_time=240):
    output.write_state(state)

# Close the output file
output.close()
