#!bin/bash
set -eux
gcc -O3 -o src/anemoi/inference/C/flatten.so -shared -fPIC -fopenmp src/anemoi/inference/C/flatten.cpp
