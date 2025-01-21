####################
 Parallel Inference
####################

If the memory requirements of your model are too large to fit within a
single GPU, you can run Anemoi-Inference in parallel across multiple
GPUs.

Parallel inference requires SLURM to launch the parallel processes and
to determine information about your network environment. If SLURM is not
available to you, please create an issue on the Anemoi-Inference github
page `here <https://github.com/ecmwf/anemoi-inference/issues>`_.

***************
 Configuration
***************

To run in parallel, you must add '`runner:parallel`' to your inference
config file.

.. code:: yaml

   checkpoint: /path/to/inference-last.ckpt
   lead_time: 60
   runner: parallel
   input:
     grib: /path/to/input.grib
   output:
     grib: /path/to/output.grib

*******************************
 Running inference in parallel
*******************************

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

   If you specify '`runner:parallel`' but you don't launch with
   '`srun`', your anemoi-inference job may hang as only 1 process will
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
