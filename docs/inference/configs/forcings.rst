.. _forcings:

##########
 Forcings
##########

:ref:`inputs` refers to the input methods used to fetch the initial
conditions (prognostic variables). Anemoi-inference also supports
fetching forcings from different sources. There are two types of
forcings:

- **constant_forcings**: Variables that remain constant throughout the
  simulation (e.g., land-sea mask, orography). These are fetched once
  at initialization.

- **dynamic_forcings**: Variables that change during the forecast and
  are provided to the model at each time step (e.g., atmospheric fields
  used as forcing to an ocean model).

See :ref:`input-types` for more information on variable categories.

By default, both ``constant_forcings`` and ``dynamic_forcings`` are
fetched from the same source as ``input``. However, you can specify
different sources for each.

The example below shows how to fetch both constant and dynamic forcings
from ``mars`` while the initial conditions are fetched from ``test``:

.. literalinclude:: yaml/forcings_1.yaml
   :language: yaml

This example above is equivalent to:

.. literalinclude:: yaml/forcings_2.yaml
   :language: yaml

You can also specify different sources for each type of forcing.

To get the constant forcings from a file:

.. literalinclude:: yaml/forcings_3.yaml
   :language: yaml

To get the dynamic forcings from MARS:

.. literalinclude:: yaml/forcings_4.yaml
   :language: yaml

If ``constant_forcings`` or ``dynamic_forcings`` are not specified, they
will default to the value of ``input``.
