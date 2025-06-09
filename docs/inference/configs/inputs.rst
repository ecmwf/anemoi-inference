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

.. literalinclude:: yaml/inputs_1.yaml
   :language: yaml

``test`` is the default input if no input is specified.

If the training happened on a different computer and the datasets files
are not available on the current computer, you can use
`anemoi-datasets's` :ref:`anemoi-datasets:configuration` to define a
search path to the datasets. To enable this, you have set the
``use_original_paths`` option to ``false``.

.. literalinclude:: yaml/inputs_2.yaml
   :language: yaml

You can also provide a full dataset specification as follows:

.. literalinclude:: yaml/inputs_3.yaml
   :language: yaml

See :ref:`anemoi-datasets:opening-datasets` in the documentation of the
`anemoi-datasets` package for more information on how to open datasets.

******
 grib
******

You can specify the input as ``grib`` to read the data from a GRIB file.

.. literalinclude:: yaml/inputs_4.yaml
   :language: yaml

For more options, see :ref:`grib-input`.

****************
 icon_grib_file
****************

The ``icon_grib_file`` input is a class dedicated to reading ICON GRIB
files. It is

.. literalinclude:: yaml/inputs_5.yaml
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

.. literalinclude:: yaml/inputs_6.yaml
   :language: yaml

You can also specify some of the MARS keywords as options. The default
is to retrieve the data from the operational analysis (``class=od``).
You can change that to use ERA5 reanalysis data (``class=ea``).

.. literalinclude:: yaml/inputs_7.yaml
   :language: yaml

The ``mars`` input also accepts the ``namer`` parameter of the GRIB
input.

*****
 cds
*****

You can also specify the input as ``cds`` to read the data from the
`Climate Data Store <https://cds.climate.copernicus.eu/>`_. This
requires the `cdsapi` package to be installed, and the user to have a
CDS account.

.. literalinclude:: yaml/inputs_8.yaml
   :language: yaml

As the CDS contains a plethora of `datasets
<https://cds.climate.copernicus.eu/datasets>`_, you can specify the
dataset you want to use with the key `dataset`.

This can be a str in which case the dataset is used for all requests, or
a dict of any number of levels which will be descended based on the
key/values for each request.

You can use `*` to represent any not given value for a key, i.e. set a
dataset for `param: 2t`. and `param: *` to represent any other param.

.. literalinclude:: yaml/inputs_9.yaml
   :language: yaml

In the above example, the dataset `reanalysis-era5-pressure-levels` is
used for all with `levtype: pl` and `reanalysis-era5-single-levels` used
for all with `levtype: sfc`.

Additionally, any kwarg can be passed to be added to all requests, i.e.
for ERA5 data, `product_type: 'reanalysis'` is needed.

.. literalinclude:: yaml/inputs_10.yaml
   :language: yaml

********
 cutout
********

``cutout`` is a special type of input that combines one or more Limited
Area Model (LAM) sources into a global source using a nested cutout
approach. This is also known as the "stretched-grid" method, see Nipen
et al. (2024). The ``cutout`` input contains multiple sources, each with
its own input type (e.g. 'grib', 'mars', etc.), and the order of the
sources determines the nesting order. The first source is the innermost
domain, and the last source is the outermost, global domain,
consistently with what is done in ``anemoi-datasets``, see `here
<https://anemoi.readthedocs.io/projects/datasets/en/latest/using/combining.html#cutout>`_.

An important prerequisite is that your checkpoint must contain the
cutout masks as supporting arrays. You can check this by running the
``anemoi-inference metadata --supporting-arrays <your_checkpoint>``
command. You should be able to see some cutout masks in the output:

.. code:: output

   lam_0/cutout_mask: shape=(226980,) dtype=bool
   global/cutout_mask: shape=(542080,) dtype=bool

If these are not present, you can try to add them to your checkpoint by
running ``anemoi-inference patch <your_checkpoint>``.

An example configuration for the ``cutout`` input is shown below:

.. literalinclude:: yaml/inputs_11.yaml
   :language: yaml

The different sources are specified exactly as you would for a single
source, as shown in the previous sections.
