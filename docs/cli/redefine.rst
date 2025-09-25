.. _redefine-command:

Redefine Command
===============

With this command, you can redefine the graph of a checkpoint file.
This is useful when you want to change / reconfigure the local-domain of a model, or rebuild with a new graph.

We should caution that such transfer of the model from one graph to
another is not guaranteed to lead to good results. Still, it is a
powerful tool to explore generalisability of the model or to test
performance before starting fine tuning through transfer learning.

This will create a new checkpoint file with the updated graph, and optionally save the graph to a file.

Subcommands allow for a graph to be made from a lat/lon coordinate file, bounding box, or from a defined graph config.

*********
 Usage
*********

.. code-block:: bash

    % anemoi-inference redefine --help

    Redefine the graph of a checkpoint file.

    positional arguments:
      path                  Path to the checkpoint.

    options:
      -h, --help            show this help message and exit
      -g GRAPH, --graph GRAPH
                            Path to graph file to use
      -y GRAPH_CONFIG, --graph_config GRAPH_CONFIG
                            Path to graph config to use
      -ll LATLON, --latlon LATLON
                            Path to coordinate npy, should be of shape (N, 2) with latitudes and longitudes.
      -c COORDS COORDS COORDS COORDS COORDS, --coords COORDS COORDS COORDS COORDS COORDS
                            Coordinates, (North West South East Resolution).
      -gr GLOBAL_RESOLUTION, --global_resolution GLOBAL_RESOLUTION
                            Global grid resolution required with --coords, (e.g. n320, o96).
      --save-graph SAVE_GRAPH
                            Path to save the updated graph.
      --output OUTPUT       Path to save the updated checkpoint.


*********
Examples
*********

Here are some examples of how to use the `redefine` command:

#. Using a graph file:

    .. code-block:: bash

          anemoi-inference redefine path/to/checkpoint --graph path/to/graph

#. Using a graph configuration:

    .. code-block:: bash

          anemoi-inference redefine path/to/checkpoint --graph_config path/to/graph_config

    .. note::
        The configuration of the existing graph can be found using:

        .. code-block:: bash

            anemoi-inference metadata path/to/checkpoint -get config.graph ----yaml

#. Using latitude/longitude coordinates:
    This lat lon file should be a numpy file of shape (N, 2) with latitudes and longitudes.

    It can be easily made from a list of coordinates as follows:

    .. code-block:: python

          import numpy as np
          coords = np.array(np.meshgrid(latitudes, longitudes)).T.reshape(-1, 2)
          np.save('path/to/latlon.npy', coords)

    Once created,

    .. code-block:: bash

          anemoi-inference redefine path/to/checkpoint --latlon path/to/latlon.npy

#. Using bounding box coordinates:

    .. code-block:: bash

          anemoi-inference redefine path/to/checkpoint --coords North West South East Resolution

    i.e.

    .. code-block:: bash

          anemoi-inference redefine path/to/checkpoint --coords 30.0 -10.0 20.0 0.0 0.1/0.1 --global_resolution n320


All examples can optionally save the updated graph and checkpoint using the `--save-graph` and `--output` options.

***************************
Complete Inference Example
***************************

For this example we will redefine a checkpoint using a bounding box and then run inference


Redefine the checkpoint
-----------------------

.. code-block:: bash

    anemoi-inference redefine path/to/checkpoint --coords 30.0 -10.0 20.0 0.0 0.1/0.1 --global_resolution n320 --save-graph path/to/updated_graph --output path/to/updated_checkpoint

Create the inference config
---------------------------

If you have an input file of the expected shape handy use it in place of the input block, here we will show
how to use MARS to handle the regridding.

.. note::
    Using the `anemoi-plugins-ecmwf-inference <https://github.com/ecmwf/anemoi-plugins-ecmwf>`_ package, preprocessors are available which can handle the regridding for you from other sources.

.. code-block:: yaml

    checkpoint: path/to/updated_checkpoint
    date: -2

    input:
        cutout:
            lam_0:
                mars:
                    grid: 0.1/0.1 # RESOLUTION WE SET
                    area: 30.0/-10.0/20.0/0.0 # BOUNDING BOX WE SET, N W S E
            global:
                mars:
                    grid: n320 # GLOBAL RESOLUTION WE SET


Run inference
-----------------

.. code-block:: bash

    anemoi-inference run path/to/updated_checkpoint


**********
Reference
**********

.. argparse::
    :module: anemoi.inference.__main__
    :func: create_parser
    :prog: anemoi-inference
    :path: redefine
