.. _api_level1:

####################
 NumPy to NumPy API
####################

The simplest way to run a inference from a checkpoint is to provide the
initial state as a dictionary containing NumPy arrays for each input
variable.

You then create a Runner object and call the `run` method, which will
yield the state at each time step. Below is a simple code example to
illustrate this:

.. literalinclude:: code/level1_1.py
   :language: python

The field names are the one that where provided when running the
:ref:`training <anemoi-training:index-page>`, which were the name given
to fields when creating the training :ref:`dataset
<anemoi-datasets:index-page>`.

********
 States
********

A `state` is a Python :py:class:`dictionary` with the following keys:

-  ``date``: :py:class:`datetime.datetime` object that represent the
   date at which the state is valid.
-  ``latitudes``: a NumPy array with the list of latitudes that matches
   the data values of fields
-  ``longitudes``: a NumPy array with the corresponding list of
   longitudes. It must have the same size as the latitudes array.
-  ``fields``: a :py:class:`dictionary` that maps fields names with
   their data.

Each field is given as a NumPy array. If the model is
:py:attr:`multi-step
<anemoi.models.interface.AnemoiModelInterface.multi_step>`, it will
needs to be initialised with fields from two or more dates, the values
must be two dimensions arrays, with the shape ``(number-of-dates,
number-of-grid-points)``, otherwise the values can be a one dimension
array. The first dimension is expected to represent each date in
ascending order, and the ``date`` entry of the state must be the last
one.

As it iterates, the model will produce new states with the same format.
The ``date`` will represent the forecasted date, and the fields would
have the forecasted values as NumPy array. These arrays will be of one
dimensions (the number of grid points), even if the model is multi-step.

*************
 Checkpoints
*************

Some newer version of :ref:`anemoi-training
<anemoi-training:index-page>` will store the `latitudes` and
`longitudes` used during training into the checkpoint. The example code
above can be simplified as follows:

.. literalinclude:: code/level1_2_.py
   :language: python
