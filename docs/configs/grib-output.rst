.. _grib-output:

#############
 GRIB output
#############

.. warning::

   The GRIB output will be more efficient in conjunction with a
   :ref:`grib-input`. The GRIB output will use its input as template for
   encoding GRIB messages. If the input is not a GRIB file, the output
   will still attempt to encode the data as GRIB messages, but this is
   not always possible.
