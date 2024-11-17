.. _inputs:

########
 Inputs
########

**********
 Datasets
**********

You can use the dataset that was used during training as input. It can
be ``test``, ``training`` or ``validation``, corresponding to the
entries given during training as ``dataloader.test``,
``dataloader.training`` and ``dataloader.validation`` respectively.

.. literalinclude:: inputs_1.yaml
   :language: yaml

``test`` is the default input if no input is specified.

If the training happened on a different computer and the datasets files
are not available on the current computer, you can use
`anemoi-datasets's` :ref:`anemoi-datasets:configuration` to define a
search path to the datasets. To enable this, you have set the
``use_original_paths`` option to ``false``.

.. literalinclude:: inputs_2.yaml
   :language: yaml

You can also provide a full dataset specification as follows:

.. literalinclude:: inputs_3.yaml
   :language: yaml

See :ref:`anemoi-datasets:opening-datasets` in the documentation of the
`anemoi-datasets` package for more information on how to open datasets.

******
 grib
******

You can specify the input as ``grib`` to read the data from a GRIB file.

.. literalinclude:: inputs_4.yaml
   :language: yaml

For more options, see :ref:`grib-input`.

******
 mars
******

You can also specify the input as ``mars`` to read the data from ECMWF's
MARS archive. This requires the `ecmwf-api-client` package to be
installed, and the user to have an ECMWF account.

.. code:: yaml

   input: mars
