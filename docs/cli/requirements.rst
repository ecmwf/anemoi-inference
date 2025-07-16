.. _requirements_command:

Requirements Command
================

This command will create a Python's requirements.txt file based version of the packages that were use during training.

*********
 Usage
*********


.. argparse::
    :module: anemoi.inference.__main__
    :func: create_parser
    :prog: anemoi-inference
    :path: requirements
