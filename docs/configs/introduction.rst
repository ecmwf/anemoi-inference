##############
 Introduction
##############

This document provides an overview of the configuration options
available to control ``anemoi-inference run``.

The configuration file is a YAML file that specifies the configuration
options.

***********
 Inference
***********

checkpoint:
===========

The only compulsory option is ``checkpoint``, which specifies the path
to the checkpoint file.

.. code:: yaml

   checkpoint: /path/to/checkpoint.ckpt

date:
=====

The date option specifies the reference date of the forecast (the
starting date)

lead_time:
==========

The lead_time option specifies the forecast lead time in hours.

allow_nans:
===========

The allow_nans option specifies whether to allow NaNs in the input and
output. It set to ``null`` (default), the value is set internally to
``true`` if any of the input fields contain NaNs, otherwise it is set to
``false``. You can override thise behaviour by setting it to ``true`` or
``false`` in the configuration file.

***********
 Debugging
***********

..code:: yaml

   verbosity: 2 report_errors: True
