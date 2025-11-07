####################
 Parallel Inference
####################

.. contents:: Table of Contents
   :local:
   :depth: 2

If the memory requirements of your model are too large to fit within a
single device, you can run Anemoi-Inference in parallel across multiple
devices. The parallel runner distributes the model across devices and
coordinates inference execution.

***************
 Prerequisites
***************

Parallel inference requires:

-  Anemoi-Models >= v0.4.2 (for model parallelism support)
-  Multiple devices available on your system or cluster

.. note::

   If updating to Anemoi-Models v0.4.2 breaks your existing checkpoints,
   you can cherry-pick `the relevant PR
   <https://github.com/ecmwf/anemoi-core/pull/77>`_ into your old
   version of Anemoi-Models.

***************
 Configuration
***************

To run in parallel, add ``runner: parallel`` to your inference config
file. The parallel runner will automatically detect your cluster
environment (Slurm, MPI, torchrun, etc.) and configure itself
accordingly.

Basic Configuration
===================

For environments with automatic cluster detection (Slurm, MPI,
torchrun), a minimal configuration is sufficient:

.. code:: yaml

   checkpoint: /path/to/inference-last.ckpt
   lead_time: 60
   runner: parallel

   input:
     grib: /path/to/input.grib
   output:
     grib: /path/to/output.grib

Supported Cluster Types
=======================

The following cluster types are automatically detected:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   -  -  Cluster Type
      -  Detection Method
      -  Environment Variables Used

   -  -  **Slurm**
      -  Presence of ``SLURM_NTASKS`` and ``SLURM_JOB_NAME``
      -  ``SLURM_PROCID``, ``SLURM_LOCALID``, ``SLURM_NTASKS``,
         ``SLURM_NODELIST``

   -  -  **MPI**
      -  Presence of ``OMPI_COMM_WORLD_SIZE`` or ``PMI_SIZE``
      -  ``OMPI_COMM_WORLD_RANK``, ``OMPI_COMM_WORLD_LOCAL_RANK``,
         ``OMPI_COMM_WORLD_SIZE``

   -  -  **Distributed (torchrun)**
      -  Presence of ``RANK`` and ``LOCAL_RANK``
      -  ``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE``, ``MASTER_ADDR``,
         ``MASTER_PORT``

Manual Cluster Configuration
============================

If you are running in an environment without automatic detection, use
the manual cluster
(:class:`anemoi.inference.clusters.manual.ManualCluster`) by specifying
the cluster as ``manual`` and the ``world_size`` (number of devices):

.. code:: yaml

   checkpoint: /path/to/inference-last.ckpt
   lead_time: 60
   runner:
      parallel:
         cluster:
            manual: 4  # Use 4 devices

   input:
     grib: /path/to/input.grib
   output:
     grib: /path/to/output.grib

.. warning::

   The ``world_size`` cannot exceed the number of available devices on
   your system.

Custom Cluster Mapping
======================

Additionally, if you have a custom cluster environment, you can specify
your own environment variable mapping:

.. code:: yaml

   checkpoint: /path/to/inference-last.ckpt
   lead_time: 60
   runner:
      parallel:
         cluster:
            custom:
               mapping:
                  local_rank: LOCAL_RANK_ENV_VAR
                  global_rank: GLOBAL_RANK_ENV_VAR
                  world_size: WORLD_SIZE_ENV_VAR
                  master_addr: MASTER_ADDR_ENV_VAR
                  master_port: MASTER_PORT_ENV_VAR
                  init_method: env://

   input:
     grib: /path/to/input.grib
   output:
     grib: /path/to/output.grib

Base Runner
-----------

By default, the `parallel` runner inherits from the `default` runner
(:class:`anemoi.inference.runners.default.DefaultRunner`). If you want
to run a different runner in parallel, you can pass the ``base_runner``
option:

.. code:: yaml

   runner:
     parallel:
       base_runner: my-custom-runner

Any additional options passed to the `parallel` runner will be forwarded
to the ``base_runner``.

*******************************
 Running Inference in Parallel
*******************************

Once you have configured ``runner: parallel`` in your config file, you
can launch parallel inference by calling ``anemoi-inference run
config.yaml`` as normal.

If you are using a cluster manager like Slurm or MPI, you must launch
your job using the appropriate launcher (``srun``, ``mpirun``, etc). See
the examples below.

Parallel with Slurm
===================

Below is an example SLURM batch script to launch a parallel inference
job across 4 GPUs.

.. code:: bash

   #!/bin/bash
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=4
   #SBATCH --gpus-per-node=4
   #SBATCH --cpus-per-task=8
   #SBATCH --time=0:05:00
   #SBATCH --output=outputs/parallel_inf.%j.out

   source /path/to/venv/bin/activate
   srun anemoi-inference run parallel.yaml

.. warning::

   If you specify ``runner: parallel`` but don't launch with ``srun``,
   your anemoi-inference job may hang as only 1 process will be
   launched.

.. note::

   By default, anemoi-inference will determine your system's master
   address and port automatically. If this fails (e.g., when running
   inside a container), you can set these values manually via
   environment variables in your SLURM batch script:

   .. code:: bash

      MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
      export MASTER_ADDR=$(nslookup $MASTER_ADDR | grep -oP '(?<=Address: ).*')
      export MASTER_PORT=$((10000 + RANDOM % 10000))

      srun anemoi-inference run parallel.yaml

Parallel with MPI
=================

To run parallel inference with MPI, use ``mpirun`` or ``mpiexec`` to
launch your job:

.. code:: bash

   #!/bin/bash
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=4
   #SBATCH --gpus-per-node=4
   #SBATCH --cpus-per-task=8
   #SBATCH --time=0:05:00
   #SBATCH --output=outputs/parallel_inf_mpi.%j.out

   source /path/to/venv/bin/activate

   # Set master address and port for communication
   MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
   export MASTER_ADDR=$(nslookup $MASTER_ADDR | grep -oP '(?<=Address: ).*')
   export MASTER_PORT=29500

   mpirun -np 4 anemoi-inference run parallel.yaml

.. note::

   If your torch supports it (PyTorch must be compiled from source with
   MPI support to use the MPI backend to torch.distributed), you can use
   the ``mpi`` torch backend by configuring:

   .. code:: yaml

      runner:
         parallel:
            cluster:
               mpi:
                  use_mpi_backend: true

Parallel with torchrun
======================

For environments without a cluster manager, you can use PyTorch's
``torchrun`` utility:

.. code:: bash

   #!/bin/bash

   source /path/to/venv/bin/activate

   torchrun --nproc_per_node=4 \
            --nnodes=1 \
            --node_rank=0 \
            --master_addr=localhost \
            --master_port=29500 \
            $(which anemoi-inference) run parallel.yaml

.. note::

   When using ``torchrun``, the distributed environment variables
   (``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE``, etc.) are automatically
   set by torchrun.

***********************
 Environment Variables
***********************

The following environment variables can be used to customise parallel
inference:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   -  -  Environment Variable
      -  Description

   -  -  ``ANEMOI_BASE_SEED``

      -  Base seed for reproducible inference. Will be broadcast from
         rank 0 to all ranks. Values < 1000 are automatically multiplied
         by 1000.

*****************
 Troubleshooting
*****************

Common Issues
=============

.. list-table::
   :header-rows: 1
   :widths: 30 70

   -  -  Issue
      -  Solution

   -  -  **Job hangs indefinitely**

      -  Ensure you're launching with the appropriate launcher
         (``srun``, ``mpirun``, ``torchrun``). Check that the number of
         processes matches your configuration.

   -  -  **"No suitable cluster found" error**
      -  Add explicit cluster configuration using ``cluster: manual`` or
         verify your environment variables are set correctly.

   -  -  **Version compatibility error**
      -  Upgrade to Anemoi-Models >= v0.4.2 or cherry-pick the `parallel
         inference PR <https://github.com/ecmwf/anemoi-core/pull/77>`_.

   -  -  **CUDA out of memory**

      -  Increase the number of devices (``world_size``) to distribute
         the model across more devices. Or, increase the chunking with
         ``ANEMOI_INFERENCE_NUM_CHUNKS``.

   -  -  **Port already in use**
      -  Set ``MASTER_PORT`` to a different port number, or let Slurm
         auto-generate one.

   -  -  **Communication timeout**
      -  Check firewall settings and ensure all nodes can communicate.
         Verify ``MASTER_ADDR`` is accessible from all ranks.

Verification Checklist
======================

Before running parallel inference, verify:

#. ✓ Anemoi-Models version >= v0.4.2
#. ✓ Multiple GPUs available (``nvidia-smi`` or equivalent)
#. ✓ Configuration includes ``runner: parallel``
#. ✓ Using appropriate launcher (``srun``, ``mpirun``, or ``torchrun``)
#. ✓ Number of processes matches available devices
#. ✓ Network connectivity between nodes (multi-node only)

Expected Output
===============

When parallel inference runs successfully, you should see log messages
indicating:

-  Cluster type detected (e.g., "Using compute client: SlurmCluster")
-  Rank information (e.g., "rank00", "rank01", etc.)
-  Model loading on each rank
-  Inference progress from rank 0 (master)

Only rank 0 produces output files; other ranks assist with computation.
