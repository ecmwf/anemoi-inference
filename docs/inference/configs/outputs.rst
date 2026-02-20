.. _outputs:

#########
 Outputs
#########

*********************
 Common Parameters
*********************

Most output types support a common set of parameters that control which
variables are written, how they are processed, and when they are written.

variables
=========

The ``variables`` parameter allows you to subset the output to only include
specific variables. This is useful when you only need a subset of the model
output, reducing output file size and processing time.

.. literalinclude:: yaml/outputs_variables.yaml
   :language: yaml

The variable names should match the names as they appear in the checkpoint.
For variables with pressure levels, use the format ``{param}_{level}``, for
example ``t_850`` for temperature at 850 hPa.

post_processors
===============

Output-level post-processors can be applied to modify the data before it is
written. These are applied after top-level post-processors. See
:ref:`inference-processors` for details on available post-processors.

.. literalinclude:: yaml/outputs_post_processors.yaml
   :language: yaml

Common use cases include:

- Extracting subsets with ``extract_mask``, ``extract_slice``, or ``extract_from_state``
- Applying backward transforms with ``backward_transform_filter``
- Accumulating fields with ``accumulate_from_start_of_forecast``
- Assigning data to masked regions with ``assign_mask``

Extracting Subsets
-------------------

Several post-processors are available for extracting subsets of the data:

**extract_mask**
  Extracts a subset of points using a boolean mask. The mask can be provided
  as a supporting array in the checkpoint or loaded from a file.

  .. code:: yaml

     output:
       netcdf:
         path: /path/to/output.nc
         post_processors:
           - extract_mask: thinning

**extract_slice**
  Extracts a subset of points using a slice notation.

  .. code:: yaml

     output:
       netcdf:
         path: /path/to/output.nc
         post_processors:
           - extract_slice: "::2"  # Every other point

**extract_from_state**
  Extracts a subset of points based on masks included in the state. Must be
  used with the ``Cutout`` input. This is particularly useful for extracting
  Limited Area Model (LAM) domains from global forecasts.

  .. code:: yaml

     output:
       netcdf:
         path: /path/to/lam.nc
         post_processors:
           - extract_from_state: lam_0

  For more information on using ``extract_from_state`` with the ``Cutout``
  input, see :ref:`inference-inputs`.

**assign_mask**
  Assigns the state to a larger array using a mask. This is the opposite of
  ``extract_mask`` - instead of extracting a smaller area, it assigns data
  to a portion of a larger area. Areas not covered by the mask are filled
  with a specified value (NaN by default).

  .. code:: yaml

     output:
       netcdf:
         path: /path/to/output.nc
         post_processors:
           - assign_mask:
               mask: source0/trimedge_mask
               fill_value: .nan

output_frequency
================

The ``output_frequency`` parameter controls how often output is written. By
default, output is written at every forecast step. You can specify a different
frequency using a time interval.

.. literalinclude:: yaml/outputs_frequency.yaml
   :language: yaml

The frequency can be specified in hours (e.g., ``6h``, ``12h``) or as an
integer representing hours.

write_initial_state
===================

The ``write_initial_state`` parameter controls whether the initial state
(step 0) is written to the output. By default, this is inherited from the
top-level configuration.

.. literalinclude:: yaml/outputs_write_initial.yaml
   :language: yaml

Set to ``true`` to include the initial state, or ``false`` to start from the
first forecast step.

Combining Parameters
====================

These parameters can be combined to create sophisticated output configurations:

.. literalinclude:: yaml/outputs_combined.yaml
   :language: yaml

*********
 Printer
*********

This is the default output. It prints the values minimum and maximum
values of a few fields for each forecasting time step. It does not take
any parameters.

.. literalinclude:: yaml/outputs_1.yaml
   :language: yaml

``printer`` is the default output if no output is specified.

Example of output:

.. code:: console

   😀 date=2022-09-11T00:00:00 latitudes=(40320,) longitudes=(40320,) fields=88

    q_50   shape=(40320,) min=7.90953e-07    max=3.18848e-06
    t_700  shape=(40320,) min=219.916        max=289.703
    v_300  shape=(40320,) min=-39.5658       max=58.6892
    z_100  shape=(40320,) min=139951         max=164233
    tp     shape=(40320,) min=0              max=0.897616

******
 grib
******

The `grib` output writes the output to a GRIB file.

.. literalinclude:: yaml/outputs_2.yaml
   :language: yaml

The path to the output file is actually a template.

.. code:: yaml

   path: "output-{date}-{time}-{step}.grib"

For each field, the output path will be constructed by substituting the
string between curly braces with the corresponding value, based on the
GRIB's eccodes keys. Optionally, string format specifiers can be used to
format the values. For example,

.. code:: yaml

   path: "output-{date}-{time}-{step:03}.grib"

will apply zero-padding to the 'step' value, so that it is always 3
digits long.

It relies heavily on having a GRIB input, but will nevertheless attempt
to encode the data as GRIB messages when this is not the case. For more
information, see :ref:`grib-output`.

The grib output supports all common parameters including ``variables``,
``post_processors``, ``output_frequency``, and ``write_initial_state``.

********
 netcdf
********

The `netcdf` output writes the output to a NetCDF file. The encoding is
basic, and the code does not attempt rebuild 3D fields from 2D fields.
The only coordinates are `latitude`, `longitude` and `time`. Each state
variable is written in its own NetCDF variable.

.. literalinclude:: yaml/outputs_3.yaml
   :language: yaml

The netcdf output supports all common parameters including ``variables``,
``post_processors``, ``output_frequency``, and ``write_initial_state``.

******
 zarr
******

The `zarr` output writes the output to a Zarr file. The encoding is
basic, and does not attempt to rebuild 3D fields from 2D fields. The
only coordinates are `latitude`, `longitude` and `time`.

Each state variable is written in its own Zarr array.

.. literalinclude:: yaml/outputs_10.yaml

The zarr output supports all common parameters including ``variables``,
``post_processors``, ``output_frequency``, and ``write_initial_state``.

******
 plot
******

The ``plot`` output writes the output to a series of plots. The plots
are produced in a directory, and the file are name according to a
template. They are produced with `earthkit-plots`. You can specify which
variables are plotted, and the domain to be shown.

.. literalinclude:: yaml/outputs_4.yaml
   :language: yaml

The plot output supports all common parameters including ``variables``,
``post_processors``, ``output_frequency``, and ``write_initial_state``.
The ``variables`` parameter is particularly useful for the plot output to
limit which fields are visualised.

*****
 raw
*****

The raw output writes the state variables collection of ``npz`` files in
the given directory.

.. literalinclude:: yaml/outputs_5.yaml
   :language: yaml

.. note::

   The raw output is useful for debugging and testing purposes. If it is
   proven useful, it can be developed further.

The raw output supports all common parameters including ``variables``,
``post_processors``, ``output_frequency``, and ``write_initial_state``.

*****
 tee
*****

The ``tee`` output writes the state to several outputs at once.

.. literalinclude:: yaml/outputs_6.yaml
   :language: yaml

Each output in the ``tee`` can have its own ``variables``,
``post_processors``, ``output_frequency``, and ``write_initial_state``
parameters. This allows you to write different subsets of variables to
different outputs or apply different post-processing to each output.

Here's an advanced example showing different options for each output:

.. literalinclude:: yaml/outputs_tee_advanced.yaml
   :language: yaml

*******
 truth
*******

For use with forecasts taking place in the past, the ``truth`` output
will write out data from an ``input`` class alongside the
``predictions``.

It is best used with the ``tee`` output, as it will write out the
``truth`` data to a separate file. It is capable of using any of the
outputs that are available to the ``output`` class.

.. literalinclude:: yaml/outputs_9.yaml

The nested output within ``truth`` supports all common parameters including
``variables``, ``post_processors``, ``output_frequency``, and
``write_initial_state``.
