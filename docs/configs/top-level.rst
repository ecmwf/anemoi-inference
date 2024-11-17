###################
 Top-level options
###################

***********
 Inference
***********

The following options control the inference process:

checkpoint:
===========

The only compulsory option is ``checkpoint``, which specifies the path
to the checkpoint file.

.. code:: yaml

   checkpoint: /path/to/checkpoint.ckpt

device:
=======

The ``device`` option specifies the device to use for inference.

precision:
==========

The ``precision`` option specifies the precision to use for inference.

date:
=====

The ``date`` option specifies the reference date of the forecast (the
starting date)

lead_time:
==========

The ``lead_time`` option specifies the forecast lead time in hours.

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

The ``report_errors`` option specifies whether to report errors. It is
set to ``false`` by default.

***************
 Miscellaneous
***************

use_grib_paramid:
=================

The ``use_grib_paramid`` option specifies whether to use the eccodes
paramId instead of parameter names when appropriate. It is set to
``false`` by default.
