.. _input-types:

###############################
 Input types
###############################

Anemoi-inference allows you to specify different input for different type of variables. variables
are classified using these categories:


- **computed**: Variables that are calculated during the model run.
- **forcing**: Variables that are imposed on the model from external sources.
- **prognostic**: Variables that are both input (initial conditions) and output.
- **diagnostic**: Variables that are only output, derived from other variables.
- **constant**: Variables that remain unchanged throughout the simulation, such as static fields or parameters.
- **accumulation**: Variables that represent accumulated quantities over time, such as total precipitation.

To find out which category a variable belongs to, you can use the :ref:`inspect-command` command.


The runner has now three inputs:

input: used to fetch the prognostics for the initial conditions (e.g. 2t in an atmospheric model).

constant_forcings: used to fetch the constants for the initial conditions (e.g. lsm or orography)

dynamic_forcings: used to fetch the forcings needed be some models throughout the length of the forecast (e.g. atmospheric fields used as forcing to an ocean model)

To ensure backward compatibility, unless given explicitly in the config, constant_forcings and dynamic_forcings both fallback to the input entry.

A new config option lets the user select which category of variables are written to the output if write_initial_condistions is true. For backward compatibility, it defaults to prognotic and constant_forcings
