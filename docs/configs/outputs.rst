.. _outputs:

#########
 Outputs
#########

*********
 Printer
*********

This is the default output. It prints the values minimum and maximum
values of a few fields for each forecasting time step. It does not take
any parameters.

.. literalinclude:: outputs_1.yaml
   :language: yaml

``printer`` is the default output if no output is specified.

******
 grib
******

The `grib` output writes the output to a GRIB file.

.. literalinclude:: outputs_2.yaml
   :language: yaml

It relies heavily on having a GRIB input, but will nevertheless attempt
to encode the data as GRIB messages when this is not the case. For more
information, see :ref:`grib-output`.

********
 netcdf
********

The `netcdf` output writes the output to a NetCDF file. The encoding is
basic, and the code does not attempt rebuild 3D fields from 2D fields.
The only coordinates are `latitude`, `longitude` and `time`. Each state
variable is written in its own NetCDF variable.

.. literalinclude:: outputs_3.yaml
   :language: yaml

******
 plot
******

The ``plot`` output writes the output to a series of plots. The plots
are produced in a directory, and the file are name according to a
template. Dates are formatted using the `strftime` function before being
used by the template. You can select which variables to plot.

.. literalinclude:: outputs_4.yaml
   :language: yaml

.. warning::

   This feature is experimental and is not yet fully implemented. The
   plot generated are very basic and are intended for debugging
   purposes. This feature will be developed further.

*****
 raw
*****

The raw output writes the state variables collection of ``npz`` files in
the given directory.

.. literalinclude:: outputs_5.yaml
   :language: yaml

.. note::

   The raw output is useful for debugging and testing purposes. If is
   proven useful, it can be developed further.

*****
 tee
*****

The ``tee`` output writes the state to several outputs at once.

.. literalinclude:: outputs_6.yaml
   :language: yaml

************
 apply_mask
************

The ``apply_mask`` output applies a mask to the output fields. The mask
is found in the checkpoint file. This is experimental and is intended to
be used to undo some merging of fields that was done in the input. The
result is passed to the next output.

.. literalinclude:: outputs_7.yaml

*************
 extract_lam
*************

Similar to the previous one, ``extract_lam`` will extract the LAM domain
from the output fields. The LAM domain is found in the checkpoint file,
and is based on `anemoi-datasets's` :ref:`cutout feature
<anemoi-datasets:combining-datasets>`.

.. literalinclude:: outputs_8.yaml
