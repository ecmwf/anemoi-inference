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

.. note::

   This is placeholder documentation. This will explain how to use the
   GRIB output's behaviour can be configured.

The ``grib`` output can be used to specify many more options:

.. literalinclude:: grib-output_1.yaml

***********
 encoding:
***********

A dictionary of key/value pairs to add to the encoding of the GRIB
messages.

*****************
 check_encoding:
*****************

A boolean to check that the GRIB messages have been encoded correctly.

***********
 template:
***********

If the input is not a GRIB file, the output can use an ``input`` source
to find similar fields to act as a template for encoding the GRIB.

-  ``source``: An input source to use as template for encoding the GRIB.

-  ``date``: The to use when looking for the template (default is the
   date at the output field to encode).

-  ``reuse``: A boolean to reuse the template for all fields (default is
   ``false``). If `true`, the template fetch for the first field will be
   used for all fields, irrespective of the variable to encode.

-  ``archive_requests``: This is a private feature used to generate the
   necessary information to archive the result of the run into ECMWF's
   MARS archive.
