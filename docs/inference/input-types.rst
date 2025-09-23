.. _input-types:

#############
 Input types
#############

Anemoi-inference allows you to specify different input for different
type of variables. Variables are classified using these categories:

-  ``computed``: Variables that are calculated during the model run.
-  ``forcing``: Variables that are imposed on the model from external
   sources.
-  ``prognostic``: Variables that are both input (initial conditions)
   and output.
-  ``diagnostic``: Variables that are only output, derived from other
   variables.
-  ``constant``: Variables that remain unchanged throughout the
   simulation, such as static fields or parameters.
-  ``accumulation``: Variables that represent accumulated quantities
   over time, such as total precipitation.

To find out which category a variable belongs to, you can use the
:ref:`anemoi-inference inspect <inspect-command>` command:

.. code:: console

   % anemoi-inference inspect --variables checkpoint.ckpt

The output will show something like:

.. code:: console

   100u           => diagnostic
   100v           => diagnostic
   10u            => prognostic
   10v            => prognostic
   2d             => prognostic
   2t             => prognostic
   cl             => constant, forcing
   cos_julian_day => computed, forcing
   cos_latitude   => computed, constant, forcing
   cos_local_time => computed, forcing
   cos_longitude  => computed, constant, forcing
   cp             => accumulation, diagnostic
   cvh            => constant, forcing
   cvl            => constant, forcing
   hcc            => diagnostic
   ...
   ro             => accumulation, diagnostic
   sdor           => constant, forcing
   sf             => accumulation, diagnostic
   sin_julian_day => computed, forcing
   ...
   tcw            => prognostic
   tp             => accumulation, diagnostic
   tvh            => constant, forcing
   tvl            => constant, forcing
   ...
   w_925          => prognostic
   z              => constant, forcing
   z_100          => prognostic
   z_1000         => prognostic
   ...

As shown above, some variables can belong to multiple categories.

It is possible to specify different : The runner has now three
:ref:`inputs <inputs>`, managing different categories of variables

-  ``input``: used to fetch the ``prognostics`` for the initial
   conditions (e.g. 2t in an atmospheric model).

-  ``constant_forcings``: used to fetch the constants for the initial
   conditions (e.g. lsm or orography). These are the variables that have
   ``constant`` **and** ``forcing`` in their category, and are not
   ``computed``, ``prognostic`` or ``diagnostic`` and are not
   ``computed``, ``prognostic`` or ``diagnostic``.

-  ``dynamic_forcings``: used to fetch the forcings needed be some
   models throughout the length of the forecast (e.g. atmospheric fields
   used as forcing to an ocean model). These are the variables that have
   ``forcing`` in their category, and are not ``computed`` or
   ``constant``.

To ensure backward compatibility, unless given explicitly in the config,
``constant_forcings`` and ``dynamic_forcings`` both fallback to the
``input`` entry.

A ``initial_state_categories`` configuration option lets the user select
a list of categories of variables to be written to the output if
``write_initial_conditions`` is ``true``. For backward compatibility, it
defaults to ``prognostics`` and ``constant_forcings``. In that case, the
``input`` will also fetch the constants and the forcing fields
