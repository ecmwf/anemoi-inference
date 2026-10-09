.. _dev-runners:

#################
 Runner overview
#################

The main entry point for inference is the Runner. This is the
primary manager of inputs, outputs, processing, etc, and runs the end-to-end prediction loop. 
This page will define the overall
architecture of the Runner, including advanced usage.

Runners follow the standard factory pattern of the repository (see
:ref:`dev-codebase-overview`). The standard runner is a forecaster,
called :class:`DefaultRunner
<anemoi.inference.runners.default.DefaultRunner>`. Forecasting
functionality is primarily defined in the superclass :class:`Runner
<anemoi.inference.runner.Runner>`, which is found in
``src/anemoi/inference/runner.py``. Other runners are
:class:`AutoRunnerFactory <anemoi.inference.runners.auto.AutoRunnerFactory>`,
which automatically processes checkpoints to return the correct Runner
class, and :class:`TemporalDownscalerMultiOutRunner
<anemoi.inference.runners.temporal_downscaler.TemporalDownscalerMultiOutRunner>`,
which is used for temporal downscaling.

*******************************
 Using and configuring Runners
*******************************

As a Runner is used to coordinate the processing, inputs and outputs for
all inference runs, it is highly configurable. We will primarily be
looking at the forecaster use case, but the other runners follow the
same overall pattern.

This is the initialisation of a Runner:

.. code:: python

   def __init__(self, config: RunConfiguration, *, classes: RunnerClasses | None = None) -> None:

As you can see, it takes in :class:`RunConfiguration
<anemoi.inference.config.run.RunConfiguration>`, which is the base
configuration input.

The other configuration option is :class:`RunnerClasses
<anemoi.inference.runner.RunnerClasses>`.

Both are discussed in more detail below.

***********************************
 Configuring with RunConfiguration
***********************************

``RunConfiguration`` is a Pydantic class which mirrors the top level
options in the :ref:`configuration yaml file <config_introduction>` --
so, the configuration options such as ``checkpoint``, ``input``,
``output`` are accessible via ``config.checkpoint``, ``config.input``,
or ``config.output``.

These configuration inputs are all used somewhere in Runner to create
the whole pipeline. These are covered in the :ref:`configuration
documentation <config_introduction>`, which shows how they are defined
in a yaml file. They can also be defined in code using
``RunConfiguration``:

.. code:: python

   config = RunConfiguration(checkpoint="checkpoint_path", input="grib")

All the defaults for each configuration are found in
``RunConfiguration``. The only required input is ``checkpoint``.

********************************
 Configuring with RunnerClasses
********************************

``RunnerClasses`` provide options for reading in metadata, tensor
handlers, and checkpoints. This class is defined in ``runner.py``:

.. code:: python

   class RunnerClasses(BaseModel):
       """Configurable class types used by the Runner.
       Child runners can override these with different classes.
       """

       model_config = ConfigDict(arbitrary_types_allowed=True)

       tensor_handler: type[TensorHandler] = TensorHandler
       checkpoint: type[Checkpoint] = Checkpoint
       metadata: type[Metadata] = Metadata

``RunnerClasses`` allows you to swap out some of the key internal classes used by the runner (also known as a *trait*). It consists of a set of classes which are used to initialise
the :class:`TensorHandler <anemoi.inference.tensors.TensorHandler>`,
:class:`Checkpoint <anemoi.inference.checkpoint.Checkpoint>`, and
:class:`Metadata <anemoi.inference.metadata.Metadata>`.

This facility is there to support extending the base runner to perform tasks other than forecasting, without having to rewrite or overload large parts of the runner.

By default, when creating metadata, runner passes into the ``Metadata``
class, which defines the kind of "standard" Metadata processing.

The structure of that class is:

.. code:: python

   class Metadata(LegacyMixin):
       """Base Metadata class."""

       def __init__(self, metadata: dict[str, Any], supporting_arrays: dict[str, FloatArray] = {}):
           """Initialize the Metadata object.

           Parameters
           ----------
           metadata : dict
               The metadata dictionary.
           supporting_arrays : dict, optional
               The supporting arrays, by default {}.
           """
           ...

If you needed some kind of custom metadata management in your runner,
you could subclass this class:

.. code:: python

   class MyMetadata(Metadata):
       """My custom metadata class."""

       def __init__(self, metadata: dict[str, Any], supporting_arrays: dict[str, FloatArray] = {}):
           super().__init__(metadata, supporting_arrays)
           # My custom code here

This can then be passed into the Runner class using the
``RunnerClasses`` structure:

.. code:: python

   @runner_registry.register("custom-runner")
   class MyRunner(Runner):
       def __init__(self, config: RunConfiguration):
           super().__init__(
               config,
               classes=RunnerClasses(
                   metadata=MyMetadata,
               ),
           )

This line overrides ``RunnerClasses`` to pass in the custom
``MyMetadata`` class. The rest of the defaults from ``RunnerClasses``
are still used.

It is also possible to use an existing Runner, and just override the
``RunnerClasses``:

.. code:: python

   my_runner = DefaultRunner(config, classes=RunnerClasses(metadata=MyMetadata))

*************************
 Creating custom Runners
*************************

If the configuration you need is not covered in the configuration or
the ``RunnerClasses`` options, you can create custom runners which
override runner behaviour by inheriting from ``Runner``. If creating a
custom runner, ensure that it is registered in the runner registry so
the rest of the code can access it. This can then be selected using the
:doc:`runner <../inference/configs/top-level>` configuration option.

.. code:: python

   @runner_registry.register("custom-runner")
   class MyRunner(Runner):
       def __init__(self, config: RunConfiguration):
           super().__init__(
               config,
               classes=RunnerClasses(
                   metadata=MyMetadata,
               ),
           )
