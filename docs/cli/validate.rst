.. _validate_command:

Validate Command
================

It is possible to investigate a checkpoint file and determine if the environment matches,
or if anemoi packages differ in version between the inference and the training environment.

This can be very useful to resolve issues when running an older or shared checkpoint.

*********
 Usage
*********


.. argparse::
    :module: anemoi.inference.__main__
    :func: create_parser
    :prog: anemoi-inference
    :path: validate
