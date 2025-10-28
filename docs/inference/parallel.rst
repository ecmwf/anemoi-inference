####################
 Parallel Inference
####################

If the memory requirements of your model are too large to fit within a
single GPU, you can run Anemoi-Inference in parallel across multiple
GPUs.

***************
 Prerequisites
***************

Parallel inference requires a certain minimum version of Anemoi-models
>= v0.4.2. If this breaks your checkpoints, you could cherry-pick `the
relevant PR <https://github.com/ecmwf/anemoi-core/pull/77>`_ into your
old version of Anemoi-Models.

***************
 Configuration
***************

To run in parallel, you must add '``runner:parallel``' to your inference
config file. By default it will attempt to detect your cluster
environment (e.g. Slurm, MPI, etc). If you are running on a known
cluster (e.g. Slurm), no additional configuration is required.

.. note::

   Supported cluster types which are automatically detected are:
      -  Slurm
      -  MPI
      -  Env Distributed, i.e. torchrun

If you are running in parallel without Slurm or a known cluster, you can
use the manual cluster
(:class:`anemoi.inference.clusters.manual.ManualCluster`) by setting the
cluster key in config.

Set ``world_size`` to the number of GPUs you want to use, this cannot
exceed the number of available GPUs on your system.

.. code:: yaml

   checkpoint: /path/to/inference-last.ckpt
   lead_time: 60
   runner: parallel
   cluster: # Only required if running parallel inference without a known cluster
      manual: 4
   input:
     grib: /path/to/input.grib
   output:
     grib: /path/to/output.grib

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

**************************************************
 Running inference in parallel with ManualCluster
**************************************************

Once you have added '``runner:parallel``' and configured the cluster in
your config file, you can launch parallel inference by calling
'``anemoi-inference run config.yaml``' as normal.

******************************************
 Running inference in parallel with Slurm
******************************************

Below is an example SLURM batch script to launch a parallel inference
job across 4 GPUs with SLURM.

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

   If you specify '``runner:parallel``' but you don't launch with
   '``srun``', your anemoi-inference job may hang as only 1 process will
   be launched.

.. note::

   By default, anemoi-inference will determine your systems master
   address and port itself. If this fails (i.e. when running
   Anemoi-Inference inside a container), you can instead set these
   values yourself via environment variables in your SLURM batch script:

   .. code:: bash

      MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
      export MASTER_ADDR=$(nslookup $MASTER_ADDR | grep -oP '(?<=Address: ).*')
      export MASTER_PORT=$((10000 + RANDOM % 10000))

      srun anemoi-inference run parallel.yaml
