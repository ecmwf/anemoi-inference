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
2. At each forecast step, splits the output fields into chunks according
   to the configured :ref:`chunking strategy <parallel-output-chunking>`.
3. Each chunk is dispatched to a writer process (round-robin when there
   are more chunks than writers) via a multiprocessing queue, and written
   using that writer's own instance of the wrapped output.
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

.. automethod:: anemoi.inference.outputs.parallel.ParallelOutput.__init__

.. _parallel-output-chunking:

********************
 Chunking Strategies
********************

At each forecast step, the output fields are divided into chunks that are
distributed to the writer processes. The ``chunk_strategy`` parameter
controls how this split is performed. If there are more chunks than
writers, chunks are assigned to writers round-robin
(chunk ``i`` goes to writer ``i % num_writers``).

``by_worker`` (default)
=======================

Splits the fields into exactly ``num_writers`` contiguous chunks of
roughly equal size, one per writer. This is the simplest strategy and a
good default: each writer receives a comparable amount of work.

.. code:: yaml

   output:
     parallel:
       num_writers: 4
       chunk_strategy: by_worker
       output:
         grib:
           path: forecast.grib

``by_size``
===========

Splits the fields into chunks of a fixed size, given by
``fields_per_chunk``. The number of chunks depends on the number of
output fields, and chunks are dispatched to writers round-robin. Use this
when you want to control the amount of work per queue message (e.g. to
bound memory usage), independently of ``num_writers``.

.. code:: yaml

   output:
     parallel:
       num_writers: 4
       chunk_strategy:
         by_size:
           fields_per_chunk: 10
       output:
         grib:
           path: forecast.grib

``by_metadata``
===============

Groups fields into chunks by their metadata, so that all fields sharing
the same value for the given ``keys`` are written by the same writer. The
``keys`` are read from each variable's typed metadata (e.g. ``levtype``,
``param``, ``level``). This is useful when the output backend benefits
from co-locating related fields in the same file.

.. code:: yaml

   output:
     parallel:
       num_writers: 4
       chunk_strategy:
         by_metadata:
           keys: [levtype]
       output:
         grib:
           path: forecast.grib

An optional ``max_groups`` argument caps the number of chunks produced;
if more distinct metadata groups exist than ``max_groups``, groups are
merged round-robin so that no more than ``max_groups`` chunks are created:

.. code:: yaml

   output:
     parallel:
       num_writers: 4
       chunk_strategy:
         by_metadata:
           keys: [param]
           max_groups: 8
       output:
         grib:
           path: forecast.grib

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
           store: forecast.zarr

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
dead and abort inference with a RuntimeError.
