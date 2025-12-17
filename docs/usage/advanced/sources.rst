.. _usage-advanced-sources:

####################
 Other data sources
####################

In the :ref:`usage-quickstart` guide, we showed how to use the test
dataset for initial conditions. However, in practice, you will likely
want to use live data to initialise the model.

.. note::

   Further details of the inputs available can be found in the
   :ref:`inputs module <modules-inputs>` and :ref:`input config
   <inputs>` documentation.

*************
 Local files
*************

While using the training dataset is a good way to get started, if
running in real time you will likely want more up to date data. It is
valid to regenerate the dataset with more recent data, and use that as
input. To do so, use the `dataset` input and provide the opening
configuration:

.. literalinclude:: yaml/sources1.yaml
   :language: yaml

Additionally, you can use local files directly as input with `grib`
being supported at the moment.

.. literalinclude:: yaml/sources2.yaml
   :language: yaml

***************
 Remote Stores
***************

It is also possible to use remote data stores as input sources.
Currently, ``anemoi-inference`` supports the Copernicus Data Store
(CDS), MARS and FDB as remote data sources.

.. warning::

   To download data from the Copernicus Data Store (cds), you need to
   have the ``cdsapi`` package installed and configured in your
   environment. See `here
   <https://cds.climate.copernicus.eu/how-to-api>`_ for more
   information.

.. literalinclude:: yaml/sources3.yaml
   :language: yaml

.. note::

   Usage of the ECMWF opendata service is also available, accessable
   through a plugin,

   .. code:: bash

      pip install anemoi-plugins-ecmwf-inference[opendata]

   Once the plugin is installed, you can use the `opendata` input
   source.
