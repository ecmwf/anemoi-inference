.. _inputs:

########
 Inputs
########

The input methods listed below are used to fetch the initial conditions
of a model run. They will also be the source of forcings during the
model run, unless a ``forcing`` entry is specified in the configuration
(see :ref:`forcings`).

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

****************
 icon_grib_file
****************

The ``icon_grib_file`` input is a class dedicated to reading ICON GRIB
files. It is

.. literalinclude:: inputs_5.yaml
   :language: yaml

The ``grid`` entry refers to a NetCDF file that contains the definition
of the ICON grid in radians. The ``refinement_level_c`` parameter is
used to specify the refinement level of the ICON grid. The
``icon_grib_file`` input also accepts the ``namer`` parameter of the
GRIB input.

.. note::

   Once the grids are stored by in the checkpoint `Anemoi`, the
   ``icon_grib_file`` input will become obsolete.

..
   For more options, see :ref:`icon-input`.

******
 mars
******

You can also specify the input as ``mars`` to read the data from ECMWF's
MARS archive. This requires the `ecmwf-api-client` package to be
installed, and the user to have an ECMWF account.

.. literalinclude:: inputs_6.yaml
   :language: yaml

You can also specify some of the MARS keywords as options. The default
is to retrieve the data from the operational analysis (``class=od``).
You can change that to use ERA5 reanalysis data (``class=ea``).

.. literalinclude:: inputs_7.yaml
   :language: yaml

The ``mars`` input also accepts the ``namer`` parameter of the GRIB
input.
