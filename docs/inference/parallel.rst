####################
 Parallel Inference
####################

If the memory requirements of your model are too large to fit within a
single GPU, you can run Anemoi-Inference in parallel across multiple
GPUs.

You have two options to launch parallel inference:
   -  Launch without Slurm. This allows you to run inference across
      multiple GPUs **on a single node**.
   -  Launch via Slurm. Slurm is needed to run inference **across
      multiple nodes**.

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
config file. If you are running in parallel without Slurm, you must also
add a '``world_size: num_gpus``' field. This informs Anemoi-Inference
how many GPUs you want to run across. It cannot be greater then the
number of GPUs on a single node.

.. note::

   If you are launching parallel inference via Slurm, '``world_size``'
   will be ignored in favour of the '``SLURM_NTASKS``' environment
   variable.

.. code:: yaml

   checkpoint: /path/to/inference-last.ckpt
   lead_time: 60
   runner: parallel
   world_size: 4 #Only required if running parallel inference without Slurm
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

*********************************************
 Running inference in parallel without Slurm
*********************************************

Once you have added '``runner:parallel``' and '``world_size: num_gpus``'
to your config file, you can launch parallel inference by calling
'``anemoi-inferece run config.yaml``' as normal.

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
