Introduction
============

When you install the `anemoi-inference` package, this will also install command line tool
called ``anemoi-inference`` this can be used to manage the checkpoints.

The tools can provide help with the ``--help`` options:

.. code-block:: bash

    % anemoi-inference --help

The commands are:

.. toctree::
    :maxdepth: 1

    run
    metadata
    inspect
    patch
    requests


.. argparse::
    :module: anemoi.inference.__main__
    :func: create_parser
    :prog: anemoi-inference
    :nosubcommands:
