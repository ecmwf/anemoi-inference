.. _usage-advanced-saving:

################
 Saving Outputs
################

In the :ref:`usage-quickstart` guide, we showed how to print the output
to stdout. However, in practice, you will likely want to save the output
to file.

.. note::

   Further details of the outputs available can be found in the
   :ref:`outputs module <modules-outputs>` and :ref:`output config
   <outputs>` documentation.

****************
 Output formats
****************

``anemoi-inference`` supports various output formats, including NetCDF,
GRIB, and Zarr. You can specify the desired output format in the
configuration file under the `output` section. For example, to save the
output in GRIB format, you can use the following configuration:

.. literalinclude:: yaml/saving1.yaml
   :language: yaml

.. tip::

   GRIB comes with some encoding `quirks`, so make sure to check the
   :ref:`grib output <grib-output>` documentation for more details.

To save to NETCDF for example, you can use:

.. literalinclude:: yaml/saving2.yaml
   :language: yaml
