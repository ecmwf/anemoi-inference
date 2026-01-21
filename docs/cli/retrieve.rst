.. _retrieve-command:

#################
 Retrieve Command
#################

The ``retrieve`` command generates data retrieval requests for running
inference. It analyses a checkpoint and creates MARS or JSON requests for
the required input data based on the model's configuration, lagged inputs,
and variable requirements.

************
 Description
************

The ``retrieve`` command is primarily used to prepare data for inference by
generating retrieval requests. It's particularly useful for:

- Preparing operational forecasts with specific date requirements
- Staging data from remote archives (MARS, FDB, CDS)
- Understanding what data is needed by a trained model
- Generating requests for forcing data over forecast periods

The command outputs either:

- JSON format (default): machine-readable list of requests
- MARS format (``--mars``): ready-to-use MARS retrieval commands

This command is commonly used in operational workflows where data retrieval
and model inference are separate steps.

.. argparse::
   :module: anemoi.inference.__main__
   :func: create_parser
   :prog: anemoi-inference
   :path: retrieve

*********
 Examples
*********

Generate JSON Requests for a Single Date
=========================================

.. code-block:: bash

   anemoi-inference retrieve config.yaml --date 2025-01-01T00

Output:

.. code-block:: json

   [
     {
       "class": "od",
       "stream": "oper",
       "type": "an",
       "date": "20250101",
       "time": "0000",
       "levtype": "sfc",
       "param": "2t/sp/10u/10v",
       "grid": "0.25/0.25",
       "area": "90/-180/-90/180"
     }
   ]

Generate MARS Retrieval Commands
=================================

.. code-block:: bash

   anemoi-inference retrieve config.yaml --date 2025-01-01T00 --mars

Output:

.. code-block:: text

   retrieve,
      class=od,
      stream=oper,
      type=an,
      date=20250101,
      time=0000,
      levtype=sfc,
      param=2t/sp/10u/10v,
      grid=0.25/0.25,
      area=90/-180/-90/180,
      target=input.grib

Save to File
============

.. code-block:: bash

   anemoi-inference retrieve config.yaml --date 2025-01-01T00 --output requests.json

Generate Requests for Forecast Dates (Forcings)
================================================

.. code-block:: bash

   anemoi-inference retrieve config.yaml \
       --date 2025-01-01T00 \
       --forecast-dates \
       --include forcing

This generates requests for all timesteps from the initial date through the
lead time, useful for time-varying forcing data.

Bulk Staging with Multiple Dates
=================================

.. code-block:: bash

   # Create a file with dates
   echo "2025-01-01T00" > dates.txt
   echo "2025-01-01T06" >> dates.txt
   echo "2025-01-01T12" >> dates.txt

   anemoi-inference retrieve config.yaml --staging-dates dates.txt --mars

Exclude Computed Variables
===========================

.. code-block:: bash

   anemoi-inference retrieve config.yaml \
       --date 2025-01-01T00 \
       --exclude computed,forcing

Add Extra MARS Parameters
==========================

.. code-block:: bash

   anemoi-inference retrieve config.yaml \
       --date 2025-01-01T00 \
       --extra class=ea \
       --extra expver=0001 \
       --mars

Use with Operational Inference
===============================

.. code-block:: bash

   # Step 1: Generate retrieval request
   anemoi-inference retrieve config.yaml \
       --date $(date -u +%Y-%m-%dT%H) \
       --mars > retrieve.req

   # Step 2: Retrieve data
   mars retrieve.req

   # Step 3: Run inference
   anemoi-inference run config.yaml

*********************
 Variable Categories
*********************

The following variable categories can be used with ``--include``,
``--exclude``, and ``--input-type``:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Category
     - Description

   * - ``prognostic``
     - Model prognostic variables (state variables evolved by the model)

   * - ``diagnostic``
     - Diagnostic variables (derived from prognostic state)

   * - ``constant``
     - Time-invariant fields (land-sea mask, orography, etc.)

   * - ``forcing``
     - Time-varying external forcing (solar radiation, etc.)

   * - ``computed``
     - Variables computed from other variables

***********
 Use Cases
***********

Operational Forecasting
=======================

In operational systems, data retrieval and model inference are often
separated:

1. ``retrieve`` generates the request for current conditions
2. Data is retrieved from archive
3. ``run`` executes the forecast

This separation allows for caching, parallel retrieval, and better error
handling.

Batch Processing
================

For historical reanalysis or verification, use ``--staging-dates`` to
generate requests for many dates at once.

Understanding Model Requirements
=================================

Use ``retrieve`` to inspect what data a model needs without running
inference:

.. code-block:: bash

   anemoi-inference retrieve config.yaml --date 2025-01-01T00

.. seealso::

  - :ref:`run-command` - Run inference with the retrieved data
  - :ref:`requests-command` - Show MARS requests for a running forecast
  - :ref:`config_introduction` - Configuration file reference
  - :ref:`usage-advanced-sources` - Input source documentation
