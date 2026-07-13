###################
 Top-level options
###################

***********
 Inference
***********

The following options control the inference process:

checkpoint:
===========

The only compulsory option is ``checkpoint``, which specifies the
checkpoint file. It can be a path to a local file, or a huggingface
config.

.. code:: yaml

   checkpoint: /path/to/checkpoint.ckpt

.. code:: yaml

   checkpoint:
      huggingface:
         repo_id: "ecmwf/aifs-single"
         filename: "aifs_single_v0.2.1.ckpt"

runner:
=======

The ``runner`` option selects the inference runner. The default is
``default``.

.. code:: yaml

   runner: default

Available runners:

default
-------

Runs the checkpoint model as-is. This is the standard runner for
production forecasts.

steady-state
------------

Runs the real checkpoint model but **freezes dynamic forcings** (e.g.
insolation, local solar time) at their values from the initial state.
They are not reloaded at each autoregressive step.

This is useful when combined with ``mid_processors`` that modify the
predicted fields between steps — for example, subtracting a
climatological mean tendency to isolate the anomaly response of a
perturbation:

.. code:: yaml

   runner: steady-state

   mid_processors:
     - subtract_tendency:
         tend_pl_path: /path/to/tend_pl.grib
         tend_sfc_path: /path/to/tend_sfc.grib
         param_pl: ["z", "q", "t", "u", "v", "w"]
         level_pl: [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
         param_sfc: ["msl", "10u", "10v", "2t"]

no-model
--------

Replaces the checkpoint model with a dummy that returns all-ones. This
is intended for testing pipelines (processors, inputs, outputs) without
running a real model.

parallel
--------

Distributes the model across multiple devices. See
:ref:`parallel-inference` for full documentation.

device:
=======

The ``device`` option specifies the device to use for inference. The
default is ``cuda``.

precision:
==========

The ``precision`` option specifies the precision to use for inference.
The default is taken from the checkpoint.

date:
=====

The ``date`` option specifies the reference date of the forecast (the
starting date). Is the date is not specified, the default will depend on
the selected input. For example, ``grib`` will use the date found in the
file, while ``mars`` will use yesterday's date.

Setting the date in the configuration file can be useful for debugging.
A more common use case is to set the date in the command line, as shown
in the :ref:`anemoi-inference run <run-command>` documentation.

.. code:: bash

   anemoi-inference run config.yaml date=2020-01-01T00:00:00

lead_time:
==========

The ``lead_time`` option specifies the forecast lead time in hours.
Default is 240 hours.

write_initial_state:
====================

The ``write_initial_state`` option specifies whether to write the
initial state to the output (i.e. "step zero" of the forecast). Default
is ``true``.

allow_nans:
===========

The ``allow_nans`` option specifies whether to allow NaNs in the input
and output. It set to ``null`` (default), the value is set internally to
``true`` if any of the input fields contain NaNs, otherwise it is set to
``false``. You can override this behaviour by setting it to ``true`` or
``false`` in the configuration file.


check_variables_compatibility:
==============================

By default, `Anemoi` will check that the data coming from the various inputs match the data that was used
to train the model, i.e. that the variables have the same units, same time processing, etc.

You can turn some of the checks off.

.. code:: yaml

   check_variables_compatibility:
      ignore_units: True # Don't check units
      ignore_time_processing: True # Don't check time processing (e.g. whether the data is instantaneous or accumulated)
      ignore_processing_period: True # Don't check time processing period (e.g. whether the data are 3-hourly or 6-hourly accumulations)
      ignore_type_of_level: True # Don't check type of level (e.g. whether the data are on pressure levels or model levels)


You can also turn off checks for individual variables by setting

.. code:: yaml

   check_variables_compatibility:
      ignore_type_of_level: msl # Don't check type of level for the variable "msl"
      ignore_units: [msl, t2m] # Don't check units for the variables "msl" and "t2m"

For multi-datasets models, you can specify the checks for each dataset separately:

.. code:: yaml

   check_variables_compatibility:
      dataset1:
         ignore_units: True # Don't check units for dataset1
      dataset2:
         ignore_units: [msl, t2m] # Don't check units for the variables "msl" and "t2m" in dataset2

********************
 Inputs and outputs
********************

The entries for the inputs and outputs are specified in the :ref:`inputs
<inputs>` and :ref:`outputs <outputs>` sections of the documentation.

**************************
 Pre- and Post-processors
**************************

The entries for the pre- and post-processors are specified in the
:ref:`processors <inference-processors>` section of the documentation.

***********
 Debugging
***********

verbosity:
==========

The ``verbosity`` option specifies the verbosity level of the output. It
is set to 0 by default.

use_profiler:
=============

The ``use_profiler`` option specifies whether to profile the inference
run. When enabled, the profiler produces a memory snapshot and timeline,
as well as a time summary. The profiler also adds labels to identify the
different steps of the inference to simplify visualization with Nsight.
This option is set to ``false`` by default.

***************
 Miscellaneous
***************

use_grib_paramid:
=================

The ``use_grib_paramid`` option specifies whether to use the eccodes
paramId instead of parameter names when appropriate. It is set to
``false`` by default.

env:
====

The ``env`` option specifies a dictionary of environment variables to
set before running the inference. This can be useful to set PyTorch or
OpenMP environment variables. Note that the environment variables may be
set too late in some cases.
