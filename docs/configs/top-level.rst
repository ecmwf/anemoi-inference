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
in the :ref:`anemoi-inference run <run-cli>` documentation.

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

********************
 Inputs and outputs
********************

The entries for the inputs and outputs are specified in the :ref:`inputs
<inputs>` and :ref:`inputs <outputs>` sections of the documentation.

***********
 Debugging
***********

verbosity:
==========

The ``verbosity`` option specifies the verbosity level of the output. It
is set to 0 by default.

report_errors:
==============

The ``report_errors`` option specifies whether to produce a longer error
report when the code of the model cannot be loaded. The aim od that
report is to troubleshoot versioning issues (git branches, python
modules, etc.). It is set to ``false`` by default.

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
