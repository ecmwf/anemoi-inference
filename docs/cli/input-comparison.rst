.. _input-comparison-command:

Input Comparison Command
========================

The ``input-comparison`` command loads input states from multiple configured
sources and checks them for consistency.  It is useful for diagnosing issues
such as unit mismatches, unexpected NaN patterns, or data-source divergences
before running a full forecast.

The command accepts the same YAML configuration format as the
:ref:`run command <run-command>`, with two extensions:

* ``input`` must be a **list** of input sources (rather than a single source).
* ``date`` may be a **dict** mapping each input source name to its own date,
  allowing states from different dates to be compared.

*************
 Checks
*************

For each field present in the input states the following checks are performed:

1. **NaN consistency** - the number and positions of NaN values must be the
   same across all states.
2. **Statistical consistency** (default, ``--exact`` not set):

   a. *Mean* - the mean of each state must lie within one standard deviation
      of the other states.  This primarily catches unit errors.
   b. *Standard deviation* - the relative difference between std values must
      be below ``--std-relative-threshold`` (default 0.5, i.e. 50 %).
   c. *Range* - the relative difference between ``max - min`` values must be
      below ``--range-relative-threshold`` (default 0.5, i.e. 50 %).

3. **Exact equality** (``--exact`` flag) - element-wise equality with NaNs
   treated as equal.  Only meaningful when states share the same grid and
   date.

Any inconsistency is reported as a warning.  The command exits with a
summary log message indicating whether all states are consistent.

*******************
 Plotting
*******************

When ``--plot-differences`` is set and `earthkit-plots
<https://earthkit-plots.readthedocs.io>`_ is installed, maps of the
per-field differences are saved as PNG files in the current directory.

*********
 Usage
*********

.. argparse::
    :module: anemoi.inference.__main__
    :func: create_parser
    :prog: anemoi-inference
    :path: input-comparison

***********************
 Example configuration
***********************

.. code-block:: yaml

    model: /path/to/checkpoint.ckpt

    input:
      - mars
      - validation

    date:
      mars: "2024-01-01T00:00:00"
      validation: "2024-01-01T00:00:00"

Run with:

.. code-block:: bash

    anemoi-inference input-comparison config.yaml
