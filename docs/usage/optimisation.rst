.. _usage-optimisation:

##################################
 Optimisation and Performance
##################################

This guide covers strategies for optimizing inference performance and
managing memory usage when running ``anemoi-inference``.

.. contents:: Table of Contents
   :local:
   :depth: 2

*********************
 Memory Optimisation
*********************

Large models can consume significant memory during inference. Several
strategies can help manage memory usage effectively.

Chunking
========

The most important optimisation for memory management is controlling the
chunk size for model inference. This splits the computation into smaller
batches that fit in available memory.

Environment Variable
--------------------

Set the ``ANEMOI_INFERENCE_NUM_CHUNKS`` environment variable to control
how many chunks to split each timestep into:

.. code-block:: bash

   # Split each timestep into 4 chunks
   export ANEMOI_INFERENCE_NUM_CHUNKS=4
   anemoi-inference run config.yaml

.. code-block:: bash

   # Or inline
   ANEMOI_INFERENCE_NUM_CHUNKS=8 anemoi-inference run config.yaml

.. code-block:: yaml

   # In inference config
   env:
      ANEMOI_INFERENCE_NUM_CHUNKS: 8

.. warning::

   Using too many chunks will slow down inference due to overhead.
   Start with fewer chunks and increase only if you encounter
   out-of-memory errors.

Monitoring Memory Usage
-----------------------

Monitor GPU memory during inference:

.. code-block:: bash

   # In another terminal
   watch -n 1 nvidia-smi

Look for:

- **Memory usage**: Should stay below 90% to avoid OOM
- **GPU utilization**: Should be high (> 80%) during computation
- **Fluctuations**: Large spikes may indicate inefficient chunking

Precision Reduction
===================

Using lower precision can significantly reduce memory usage with minimal
impact on forecast quality.

Half Precision (FP16)
---------------------

Most models work well with half precision:

.. code-block:: yaml

   checkpoint: /path/to/model.ckpt
   precision: 16  # Use FP16 instead of FP32
   lead_time: 240

   input:
     grib: /path/to/input.grib
   output:
     grib: /path/to/output.grib

This can reduce memory usage by approximately 50%.

BFloat16
--------

For models trained with bfloat16:

.. code-block:: yaml

   precision: bf16

.. note::

   BFloat16 is supported on newer GPUs (Ampere and later). Check your
   hardware compatibility before using this option.

Mixed Precision
---------------

The model automatically handles mixed precision computation when precision
is set to 16 or bf16. Critical operations remain in higher precision while
most computations use lower precision.

Device Selection
================

CPU Inference
-------------

For systems without GPU or when GPU memory is insufficient:

.. code-block:: yaml

   checkpoint: /path/to/model.ckpt
   device: cpu
   lead_time: 240

.. warning::

   CPU inference is significantly slower than GPU inference (typically
   10-100x). Use only when GPU is unavailable or for small models/short
   forecasts.

Specific GPU Selection
----------------------

On multi-GPU systems, select a specific device:

.. code-block:: bash

   # Use GPU 1
   CUDA_VISIBLE_DEVICES=1 anemoi-inference run config.yaml

.. code-block:: yaml

   # Or in config
   device: cuda:1

**********************************
 Profiling and Troubleshooting
**********************************

Measuring Performance
=====================

Time Individual Components
--------------------------

Add verbosity to see timing information:

.. code-block:: bash

   anemoi-inference run config.yaml --verbosity 2

Output will include timing for:

- Checkpoint loading
- Input data loading
- Each forecast step
- Output writing

Example output:

.. code-block:: text

   INFO Loading checkpoint (3.2s)
   INFO Loading input data (1.8s)
   INFO Step 1/10 (0.42s)
   INFO Step 2/10 (0.41s)
   ...
   INFO Writing output (2.1s)


Debugging Out-of-Memory Errors
===============================

If you encounter CUDA out-of-memory errors:

1. **Increase chunking:**

   .. code-block:: bash

      ANEMOI_INFERENCE_NUM_CHUNKS=8 anemoi-inference run config.yaml

2. **Reduce precision:**

   .. code-block:: yaml

      precision: 16

3. **Use parallel inference:**

   .. code-block:: yaml

      runner: parallel

4. **Check for memory leaks:**

   .. code-block:: bash

      # Monitor memory over time
      while true; do
          nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
          sleep 1
      done

5. **Clear cache:**

   .. code-block:: python

      import torch
      torch.cuda.empty_cache()

Common Issues and Solutions
============================

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Issue
     - Solution

   * - Slow first run
     - Expected for model compilation. Subsequent runs are faster.

   * - High memory usage even with chunking
     - Reduce precision to 16 or use parallel inference

   * - Low GPU utilization
     - May indicate I/O bottleneck. Use local data sources.

   * - Inference slower than expected
     - Too many chunks adds overhead. Reduce chunk count.

   * - Inconsistent timing
     - Check for background processes or thermal throttling

   * - GRIB writing slow
     - Use faster storage or write to local disk then copy

***************************
 Environment Variables
***************************

Complete list of environment variables affecting performance:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Variable
     - Default
     - Description

   * - ``ANEMOI_INFERENCE_NUM_CHUNKS``
     - 1
     - Number of chunks per timestep for memory management

   * - ``ANEMOI_BASE_SEED``
     - Random
     - Base seed for reproducibility (parallel inference)

   * - ``CUDA_VISIBLE_DEVICES``
     - All GPUs
     - Which GPUs are visible to the process

   * - ``PYTORCH_CUDA_ALLOC_CONF``
     - Default
     - PyTorch CUDA memory allocator configuration

Example Usage
=============

.. code-block:: bash

   export ANEMOI_INFERENCE_NUM_CHUNKS=4
   export CUDA_VISIBLE_DEVICES=0,1
   anemoi-inference run config.yaml

.. seealso::

   - :ref:`parallel-inference` - Distribute models across multiple GPUs
   - :ref:`usage-environment` - Environment setup and dependencies
   - :ref:`run-command` - CLI options for the run command
   - :ref:`retrieve-command` - Pre-fetch data for faster inference
