.. _usage-advanced-sources:

####################
 Other data sources
####################

.. warning::

   To download data from the Copernicus Data Store (cds), you need to
   have the ``cdsapi`` package installed and configured in your
   environment. See `here
   <https://cds.climate.copernicus.eu/how-to-api>`_ for more
   information.

.. admonition:: OpenData

   As CDS requires an account, some users may find it easier to use the
   `opendata` service from ECMWF to initialise the model. This data is
   openly available for the last three days under a permissive license.

   To install the plugin which provides access to the `opendata`
   service:

   .. code:: bash

      pip install anemoi-plugins-ecmwf-inference[opendata]

   To use this input simply reference it by name in the input block:

   .. code:: yaml

      input: opendata

   .. warning::

      Initial conditions are only available for the past three days
      using the `opendata` service.
