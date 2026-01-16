.. _dev-integration-tests:

###################
 Integration tests
###################

Anemoi-inference includes integration tests that can be executed locally
using pytest. These are located in the ``tests/integration`` directory.
They are designed to test the end-to-end functionality of inference.
Integration tests run from real input data and produce real output data
and go through the entire inference pipeline, except for the pytorch
model.

The pytorch model is mocked with a dummy model that passes prognostic
input variables through to the output without any modification.
Diagnostics are output as all ones. In the case of multi-step input, the
last input step is copied to the output. The result is that the
prognostic variables in the output should match the last step of the
input data.

The test checkpoint does contain metadata of a real model under test. So
as far as inference is concerned, it sees a real checkpoint with real
metadata. Any code that interacts with the metadata will behave as usual
(and can be tested), and the output file will be written with the
expected variables and shapes.

To keep the tests fast, models must run at a reduced resolution. O48 or
lower is recommended.

**************
 How it works
**************

The integration test is set up as a single parameterised pytest that
runs through all the models in the ``tests/integration`` directory. Each
model has its own subdirectory with a ``config.yaml`` and
``metadata.json``.

The integration tests run as part of the regular pytest suite, or can be
individually run with ``pytest tests/integration``. For each model
folder, the test will:

#. Load the test configuration from the ``config.yaml`` file.
#. Load the metadata from the ``metadata.json`` file.
#. From S3, download any input data defined in the test configuration,
   and any supporting-arrays defined in the metadata.
#. Save a checkpoint to disk with the dummy pytorch model and the real
   metadata.
#. Run inference on the checkpoint with the input data. The inference
   config is part of the test configuration.
#. Run any check functions defined in the test configuration.

***************
 Configuration
***************

Each model is individually configured with a ``config.yaml`` file that
is parsed with OmegaConf. The top level is a list of test
configurations, each of which represents a single test case for the
model. Each entry defines the input file (to be fetched from S3), output
file, inference config, and any checks to be performed on the output
file after inference.

An example config looks like this:

.. literalinclude:: integration-example.yaml
   :language: yaml

The above is an example of a single test case ``grib-in-netcdf-out``,
but different combinations of inputs, inference configs, and checks can
be added to the list. Within the same config file, each test case should
have a unique name (names can be reused between configs). In the pytest
log, each test case will be printed as ``{model}/{test-case-name}``.

The ``input`` entry can be a single file like in the example above, or a
list of files:

.. code:: yaml

   input:
     - input-lam.grib
     - input-global.grib

These are then available in the inference config as ``${input:0}`` and
``${input:1}``.

Resolvers
=========

Since the integration test uses temporary files, the config file has the
following resolvers that will be substituted at runtime:

-  ``${checkpoint:}`` Path to the temporary fake checkpoint.

-  ``${input:}`` Path to the temporary input file, downloaded from S3
   (using the ``input`` entry in the config). If the input is a list,
   individual files can be accessed with their index, e.g.
   ``${input:0}``, ``${input:1}``, etc.

-  ``${output:}`` Path to the temporary output file after inference.

-  ``${s3:}`` S3 URL in HTTP format of the model's data folder.

Checks
======

The ``checks`` section of the test configuration is a list of functions
that will be executed after inference. It uses the same factory registry
as the inference config. Check functions are located in
``anemoi.inference.testing.checks``.

Each entry is a dictionary with a single key that is the name of the
check in the registry, and the value is a dict of kwargs to pass to the
function.

By default, the following arguments are passed to the check function:

-  ``file``: The path to the output file after inference.

-  ``expected_variables``: A list of variables that should be in the
   output file, according to the metadata.

-  ``checkpoint``: The :class:`anemoi.inference.checkpoint.Checkpoint`
   object as created by the runner during the integration test. This can
   be useful if the check needs to access any metadata.

*******************
 Adding new models
*******************

To add a new model to the integration test, you must first train a
version of that model at a reduced resolution. O32 and O48 datasets are
available in the catalogue for this purpose:

-  ``aifs-ea-an-oper-0001-mars-o32-2020-2021-6h-v1``
-  ``aifs-ea-an-oper-0001-mars-o48-2020-2021-6h-v1``

Once you have a low resolution inference checkpoint, a helper script is
available in the repo to add the model to the integration test:

.. code:: shell

   $ python anemoi-inference/tests/integration/add_new_model.py -h

   usage: add_new_model.py [-h] [--files [FILES ...]] [--overwrite] model checkpoint

   Add a new model for integration tests.
   Running this script will create a new model directory in tests/integration
   with metadata and config files, and upload necessary files to S3.

   positional arguments:
     model             Model name. Can only contain alphanumeric, underscores, hyphens, or dots
     checkpoint        Path to the inference checkpoint file

   options:
     -h, --help        show this help message and exit
     --files [FILES ...], -f [FILES ...]
                       Additional files to upload to the model directory on S3
     --overwrite, -o   Overwrite existing files
     --save-fake-checkpoint
                       Save a fake checkpoint file locally alongside the real checkpoint for testing purposes.

To run this script, S3 credentials with write access must be set up, see
`anemoi.utils.remote.s3
<https://anemoi.readthedocs.io/projects/utils/en/latest/_api/anemoi.utils.remote.html#module-anemoi.utils.remote.s3>`_
for details. Test files for each model are stored in
``s3://ml-tests/test-data/samples/anemoi-integration-tests/inference/{model}``

Run this script with the model name and checkpoint file as arguments,
and any additional files to upload to the model directory on S3 (for
example, an input.grib file). Supporting-arrays will be extracted from
the checkpoint and also uploaded to S3. Make sure to give your model a
unique and descriptive name.

You can pass ``--save-fake-checkpoint`` to the script to save a copy of
the "fake" checkpoint that is created in the integration test. This can
be useful for debugging purposes, for example you can put this fake
checkpoint in your inference config and run it manually with
``anemoi-inference run`` to simulate the integration test.

After running the script, a new directory will be created in
``tests/integration/{model}`` with the metadata and an example config
file. Modify the config file and add test cases as needed. Commit the
``config.yaml`` and ``metadata.json`` files to the repository and open a
PR.

**Note that the test data S3 bucket is public.** Any files uploaded need
to have appropricate licenses and permissions to share.

***************************
 eccodes local definitions
***************************

If the inference setup relies on local eccodes definitions, these need
to be made available during the integration tests. This is done by
setting the ``ECCODES_DEFINITION_PATH`` in the `inference config
<https://github.com/ecmwf/anemoi-inference/blob/a053a9ad34f34a32c3f59317faf52bbec3217f6b/tests/integration/meteoswiss-sgm-cosmo/config.yaml#L19>`_.

Because eccodes caches the definitions on first use, model setups that
use local definitions have to be isolated from each other during the
integration tests. There is a separate workflow
``.github/workflows/python-integration.yml`` that only runs the
integration tests for models that use local definitions. This is done
with pytest markers.

See `here
<https://github.com/ecmwf/anemoi-inference/blob/a053a9ad34f34a32c3f59317faf52bbec3217f6b/tests/integration/test_integration.py#L38-L43>`_
for an example of how to mark your model with a custom marker. This can
be done based on the model name, or a flag in the config file. The
marker also needs to be registered in the `conftest.py
<https://github.com/ecmwf/anemoi-inference/blob/test/meteoswiss-integration/tests/conftest.py>`_,
and a corresponding command line flag added to pytest.

For `example
<https://github.com/ecmwf/anemoi-inference/blob/a053a9ad34f34a32c3f59317faf52bbec3217f6b/tests/conftest.py#L23-L42>`_,
there is the ``cosmo`` marker that is triggered with the ``--cosmo``
flag in pytest, for tests that rely on cosmo local definitions. If the
flag is not present, the cosmo tests are skipped.
