.. _inference-processors:

################
 Pre-processors
################

There can be top-level pre-processors that are applied to all inputs
indifferently or input-level pre-processors that are applied per-input.

In terms of order, input-level pre-processors are applied first, then
top-level ones.

**************************
 Available pre-processors
**************************

no_missing_values
=================

Replaces NaNs with the mean.

forward_transform_filter
========================

Applies a filter from `anemoi-transform
<https://anemoi.readthedocs.io/projects/transform/en/latest/_api/transform.filters.html>`_.

**************************
 Top-level pre-processors
**************************

List top-level pre-processors in ``pre_processors``:

.. code:: yaml

   pre_processors:
     - forward_transform_filter: cos_sin_mean_wave_direction
     - no_missing_values

****************************
 Input-level pre-processors
****************************

For an input like ``cutout``, you may want to apply different
pre-processors for the different inputs. As such, some inputs (namely
grib, mars and cds) also accept a ``pre_processors`` argument

.. code:: yaml

   input:
     cutout:
       lam_0:
         grib:
           path: /path/to/local.grib
           pre_processors:
             - forward_transform_filter: remove_nans
       global:
         grib:
           path: /path/to/global.grib

#################
 Post-processors
#################

There can be top-level post-processors that are applied to all outputs
indifferently or output-level post-processors that are applied
per-output.

In terms of order, top-level post-processors are applied first, then
output-level ones.

***************************
 Available post-processors
***************************

accumulate_from_start_of_forecast
=================================

Accumulate fields from zero and return the accumulated fields. .. code::
yaml

.. code:: yaml

   post_processors:
     - accumulate_from_start_of_forecast

will accumulate the necessary fields from the checkpoint.

To specify the fields to accumulate:

.. code:: yaml

   post_processors:
     - accumulate_from_start_of_forecast:
         accumulations:
           - tp

backward_transform_filter
=========================

Applies a backward transform filter from `anemoi-transform
<https://anemoi.readthedocs.io/projects/transform/en/latest/_api/transform.filters.html>`_.

***************************
 Top-level post-processors
***************************

List top-level post-processors in ``post_processors``:

.. code:: yaml

   post_processors:
     - backward_transform_filter: cos_sin_mean_wave_direction
     - accumulate_from_start_of_forecast

******************************
 Output-level post-processors
******************************

For an output like ``tee``, you may want to apply different
post-processors for the different outputs. All output (except ``tee``
and ``truth``) accept an additional ``post_processors`` argument:

.. code:: yaml

   output:
     tee:
       - netcdf: /path/to/netcdf/file.nc
       - grib:
           path: /path/to/grib/file.grib
           post_processors:
             - backward_transform_filter: cos_sin_mean_wave_direction
