.. _usage-ensembles:

#####################################
 Seeds, ensembles and reproducibility
#####################################

Some models use random numbers when they run. This is the case for
ensemble models, such as models trained with a CRPS loss, which inject
noise into the model to produce different forecasts from the same
initial conditions. This page explains how to control the random seed
used by ``anemoi-inference``, how to produce several ensemble members,
and what to expect in terms of reproducibility.

.. note::

   The seed only matters for models that use random numbers. A
   deterministic model gives the same forecast whatever the seed is.

.. contents:: Table of Contents
   :local:
   :depth: 2

*****************
 Setting the seed
*****************

The seed is passed to ``torch.manual_seed()`` before the model is run.
It is set with the :ref:`seed <top-level-seed>` option, in the
configuration file:

.. code:: yaml

   checkpoint: /path/to/checkpoint.ckpt
   seed: 42

or on the command line, as an override of the configuration:

.. code:: bash

   anemoi-inference run config.yaml seed=42

The value of the seed is used as is: it is not modified in any way. The
seed is applied by all runners, not only by the :ref:`parallel runner
<parallel-inference>`.

.. note::

   The ``ANEMOI_BASE_SEED`` environment variable is no longer read. Use
   the ``seed`` option instead.

Default behaviour
=================

If no seed is given, a random one is chosen for each run, so two runs of
the same configuration give different results with an ensemble model.

The seed that is actually used is written to the log at the ``INFO``
level, as part of the line that reports the runner and device:

.. code:: text

   Using DefaultRunner runner, device=cuda, seed=1234567890

If you want to reproduce a run for which you did not set the seed, you
can set ``seed`` to the value found in its log.

***********************************
 Producing several ensemble members
***********************************

A run of ``anemoi-inference`` produces a single forecast, and there is
no built-in notion of ensemble member. To produce an ensemble of ``N``
members, run the model ``N`` times with a different seed each time, and
write each run to a different output.

For example, with a configuration file ``config.yaml`` that contains:

.. code:: yaml

   checkpoint: /path/to/ensemble-checkpoint.ckpt
   date: 2024-01-01T00:00:00
   lead_time: 120
   output:
      grib:
         path: member-0.grib

the following shell loop runs 10 members, using the member number as
seed:

.. code:: bash

   for member in $(seq 1 10); do
       anemoi-inference run config.yaml \
           seed=${member} \
           output.grib.path=member-${member}.grib
   done

The runs are independent, so they can also be submitted as separate
jobs, for example as a job array, with the member number taken from the
job array index.

Use a different seed for each member, and the same set of seeds if you
want to be able to reproduce the whole ensemble. Leaving the seed unset
also gives different members, but the ensemble then cannot be reproduced
unless you collect the seeds from the logs.

.. note::

   ``anemoi-inference`` does not derive the seed from a member number.
   It is up to you to choose seeds that differ between members.

Member number in the output
===========================

The output is not aware of the member number either. If you need it in
the GRIB output, it can be passed as a GRIB key with the ``encoding``
option of the :ref:`GRIB output <grib-output>`, for example:

.. code:: yaml

   output:
      grib:
         path: member-3.grib
         encoding:
            type: pf
            number: 3

Setting ``type`` to ``pf`` (perturbed forecast) or ``cf`` (control
forecast) makes the output encode the messages as ensemble messages. You
may need a suitable :ref:`template <templates>` for this to work with
your data.

**************
 Parallel runs
**************

When a model is split over several GPUs with the :ref:`parallel runner
<parallel-inference>`, the whole run still produces a single ensemble
member. The noise is generated for the whole grid on each rank, and each
rank then keeps its own part, so all the ranks must use the same seed.
This is done for you: the seed of rank 0 (the one from the configuration
if it was set, otherwise a random one) is broadcast to all the ranks.

To produce several members with the parallel runner, launch several
parallel runs with different seeds, as above.

****************
 Reproducibility
****************

In an unchanged environment (same hardware, number of GPUs, software
versions and configuration), using the same seed gives the same random
numbers, and therefore the same noise in the model, from one run to the
next. The random numbers drawn for a given seed may differ if any of
these change.

Even then, the forecasts are not guaranteed to be bit-for-bit identical:
``anemoi-inference`` does not enable the deterministic algorithms of
PyTorch, so the results can differ slightly between runs. Differences are
more likely when changing, for example:

-  the hardware, such as the type of GPU,
-  the precision used to run the model,
-  the number of GPUs used,
-  the versions of PyTorch and of the libraries it relies on.

Small differences are expected to grow with the lead time. To get results
that are as close as possible between runs, use the same seed, the same
software environment and the same hardware and number of GPUs.
