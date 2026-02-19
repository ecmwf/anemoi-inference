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
  at initialisation.

- **dynamic_forcings**: Variables that change during the forecast and
  are provided to the model at each time step (e.g., atmospheric fields
  used as forcing to an ocean model).

See :ref:`input-types` for more information on variable categories.

**********************
Default behaviour
**********************

By default, both ``constant_forcings`` and ``dynamic_forcings`` are
fetched from the same source as ``input``. If you only specify ``input``,
all variables (prognostics and forcings) will be fetched from that source:

.. literalinclude:: yaml/forcings_1.yaml
   :language: yaml

In this example, initial conditions, constant forcings, and dynamic forcings
are all fetched from ``test``.

**********************
Override a forcing
**********************

You can override the source for one type of forcing while keeping the other
using the default ``input`` source.

To fetch dynamic forcings from ``mars`` while initial conditions and constant
forcings come from ``test``:

.. literalinclude:: yaml/forcings_2.yaml
   :language: yaml

**********************
Override both forcings
**********************

You can specify different sources for both types of forcing:

.. literalinclude:: yaml/forcings_3.yaml
   :language: yaml

In this example, initial conditions come from ``test``, constant forcings
from ``mars``, and dynamic forcings from a GRIB file.

Each forcing configuration can use any of the available :ref:`inputs` methods
with their full configuration options. For example, the ``grib`` input accepts
options like file path and other parameters as shown for ``dynamic_forcings``
above.

If ``constant_forcings`` or ``dynamic_forcings`` are not specified, they
will default to the value of ``input``.
