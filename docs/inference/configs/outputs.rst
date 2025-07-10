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

********
 netcdf
********

The `netcdf` output writes the output to a NetCDF file. The encoding is
basic, and the code does not attempt rebuild 3D fields from 2D fields.
The only coordinates are `latitude`, `longitude` and `time`. Each state
variable is written in its own NetCDF variable.

.. literalinclude:: yaml/outputs_3.yaml
   :language: yaml

******
 plot
******

The ``plot`` output writes the output to a series of plots. The plots
are produced in a directory, and the file are name according to a
template. Dates are formatted using the `strftime` function before being
used by the template. You can select which variables to plot.

.. literalinclude:: yaml/outputs_4.yaml
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

.. literalinclude:: yaml/outputs_5.yaml
   :language: yaml

.. note::

   The raw output is useful for debugging and testing purposes. If is
   proven useful, it can be developed further.

*****
 tee
*****

The ``tee`` output writes the state to several outputs at once.

.. literalinclude:: yaml/outputs_6.yaml
   :language: yaml

************
 apply_mask
************

The ``apply_mask`` output applies a mask to the output fields. The mask
is found in the checkpoint file. This is experimental and is intended to
be used to undo some merging of fields that was done in the input. The
result is passed to the next output.

.. literalinclude:: yaml/outputs_7.yaml

*************
 assign_mask
*************

This operation can be seen as the opposite of `apply_mask`. Instead of
extracting a smaller area from a larger one, it assigns the current
output to a portion of a larger area using a mask. This is useful when
you want to restore the original state of the model after applying a
mask to it. The portion of the state that is not covered by the mask
will be set to a fill value (NaN by default).

.. literalinclude:: yaml/outputs_assign.yaml

*************
 extract_lam
*************

Similar to the previous one, ``extract_lam`` will extract the LAM domain
from the output fields. The LAM domain is found in the checkpoint file
as supporting array, and is based on the `cutout_mask` associated to
`anemoi-datasets's` :ref:`cutout feature
<anemoi-datasets:combining-datasets>`. The default mask that is used is
`lam_0` and does not need to be specified. In that case the config takes
the simple form

.. literalinclude:: yaml/outputs_8a.yaml

In the case of more complicated datasets, e.g. those formed using the
`join` operation, there can be multiple cutout masks present and the
relevant one needs to be specified. A more elaborate config could for
example be

.. literalinclude:: yaml/outputs_8b.yaml

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
