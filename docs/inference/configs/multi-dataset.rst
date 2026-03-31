.. _multi-dataset:

#################################
 Multi-dataset configuration
#################################

Anemoi-inference supports checkpoints trained on multiple datasets. When a multi-dataset checkpoint
is used, the runner creates independent inputs, outputs, pre-processors,
and post-processors for each dataset.

The config entries that support per-dataset configuration are:
``input``, ``constant_forcings``, ``dynamic_forcings``, ``output``,
``pre_processors``, and ``post_processors``. Each of these can
optionally be written in a multi-dataset format where entries are placed
inside a dictionary keyed by dataset name. The dataset names are defined in
the checkpoint metadata.

.. tip::

   Single-dataset checkpoints are fully backwards compatible.
   No config changes are needed. The existing config format continues to
   work as before.

.. note::

   In the examples below, we use the dataset names ``era5`` and ``cerra`` for illustration purposes.
   These must match the actual dataset names in the checkpoint metadata.
   During training, they can be freely chosen, and could be something like ``local`` and ``global`` instead.
   Refer to :ref:`anemoi-training:user-guide/multi-datasets` for more details on how to set up multi-dataset training.

*********************************
 Per-dataset config format
*********************************

To provide different configuration per dataset, nest the config under
keys matching the dataset names in the checkpoint:

.. code:: yaml

   output:
     era5:
       grib: output-era5.grib
     cerra:
       netcdf: output-cerra.nc

In this example, ERA5 writes to a GRIB file and CERRA writes to a NetCDF file.

The same pattern applies to inputs:

.. code:: yaml

   input:
     era5:
       mars:
         class: ea
     cerra:
       grib: cerra-input.grib

And to processors:

.. code:: yaml

   post_processors:
     era5:
       - accumulate_from_start_of_forecast
     cerra: []

*********************************
 Shared config
*********************************

If a config entry is **not** written in the per-dataset format, it is
reused for all datasets. For example:

.. code:: yaml

   input:
     grib: input.grib

With a multi-dataset checkpoint, this creates a separate GRIB input
for **each** dataset, and each one reads ``input.grib`` independently.

.. warning::

   Having multiple datasets read from the same input file is not
   optimised — each input opens and reads the file separately. It is
   recommended to use separate input files per dataset when possible.

The same applies to outputs and processors. If you write:

.. code:: yaml

   output: printer

Every dataset will use the ``printer`` output.

.. warning::

   You cannot mix shared and per-dataset config for the same entry. For example, this is not allowed:

   .. code:: yaml

      output:
         grib: output.grib
         era5:
            netcdf: output-era5.nc

   And when per-dataset config is used for an entry, all datasets must be specified.
   For example, if the model also expects a dataset named ``cerra``, this is not allowed:

   .. code:: yaml

      output:
         era5:
            netcdf: output-era5.nc

*********************************
 Dataset name placeholder
*********************************

Input and output path arguments support a ``{dataset}`` placeholder
that is automatically substituted with the dataset name. This provides
an easy way to create per-dataset file paths without writing out the
full per-dataset config.

For example, with a checkpoint trained on ``era5`` and ``cerra``:

.. code:: yaml

   output:
     netcdf: output-{dataset}.nc

This is equivalent to:

.. code:: yaml

   output:
     era5:
       netcdf: output-era5.nc
     cerra:
       netcdf: output-cerra.nc

*********************************
 Output path uniqueness
*********************************

When using multiple datasets, each dataset's output must write to a
different file path. The runner will raise an error if two outputs
resolve to the same path. Using the ``{dataset}`` placeholder is an easy way to ensure unique paths.
