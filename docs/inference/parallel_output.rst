.. _parallel-output:

##################
 Parallel Output
##################

.. contents:: Table of Contents
   :local:
   :depth: 2

When running inference on large models with many output fields, writing
results to disk can become a bottleneck. The **Parallel Output** wrapper
offloads the I/O work to multiple writer processes, allowing the main
inference loop to continue without waiting for disk writes to complete.

Each writer process handles a subset of the output fields and writes
them independently to its own file. This distributes both the I/O
bandwidth and the encoding/serialisation cost across multiple CPU cores.

***************
 How It Works
***************

The ``parallel`` output wraps any other output type (e.g. ``grib``,
``zarr``, ``netcdf``) and:

1. Spawns ``num_writers`` writer processes (using ``fork``).
2. At each forecast step, splits the output fields evenly across the
   writers.
3. Each writer receives its chunk via a multiprocessing queue and writes
   it using its own instance of the wrapped output.
4. On shutdown, all writers are gracefully terminated with a signal via
   their queues.

Each writer appends a suffix ``_w<id>`` to the output file name to
avoid conflicts (e.g. ``output_w0.grib``, ``output_w1.grib``).

***************
 Configuration
***************

To enable parallel output, wrap your existing output configuration
inside a ``parallel`` block:

.. code:: yaml

   output:
     parallel:
       num_writers: 4
       output:
         grib:
           path: /path/to/output.grib

This produces four output files:

- ``/path/to/output_w0.grib``
- ``/path/to/output_w1.grib``
- ``/path/to/output_w2.grib``
- ``/path/to/output_w3.grib``

This short syntax, where the inner output is implied, is also supported:
.. code:: yaml

   output:
     parallel:
       num_writers: 4
       grib:
         path: /path/to/output.grib

Parameters
==========

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   -  -  Parameter
      -  Required
      -  Description

   -  -  ``num_writers``
      -  Yes
      -  Number of writer processes to spawn. Each writer handles an
         equal share of the output fields.

   -  -  ``output``
      -  Yes
      -  The inner output configuration. Can be any supported output
         type (``grib``, ``zarr``, ``netcdf``, etc.).

Examples
========

GRIB output with 2 writers
---------------------------

.. code:: yaml

   output:
     parallel:
       num_writers: 2
       output:
         grib:
           path: forecast.grib

Zarr output with 4 writers
---------------------------

.. code:: yaml

   output:
     parallel:
       num_writers: 4
       output:
         zarr:
           path: forecast.zarr

NetCDF output with 8 writers
------------------------------

.. code:: yaml

   output:
     parallel:
       num_writers: 8
       output:
         netcdf:
           path: forecast.nc

Combining with parallel inference
-----------------------------------

Parallel output can be used together with the parallel runner for
distributed model inference. In this case, the CPU process associated with GPU 0 will handle the output writing.
Only CPU process 0 will spawn writer processes.

.. code:: yaml

   runner: parallel
   lead_time: 240h
   checkpoint: /path/to/checkpoint.ckpt

   input:
     grib: /path/to/input.grib

   output:
     parallel:
       num_writers: 4
       output:
         grib:
           path: /path/to/output.grib

***************************
 Choosing ``num_writers``
***************************

For small resolutions or infrequent output steps, parallel writing might be unnecessary.

For larger output sizes, one can set ``num_writers: 1``. This will produce a single output file, but the writing will happen asynchronously in a separate process, allowing the main inference loop to continue without waiting for disk I/O.

If you suspect that writing to disk is a bottleneck, you can experiment with increasing ``num_writers``. 2 or 4 writers is often a good starting point.

***********************
 Troubleshooting
***********************

Writer process crashes
=======================

If a writer process crashes (e.g. due to a disk-full error or a bug in
the output backend), the main process will detect that the writer is
dead and abort inerence with a RuntimeError. 
