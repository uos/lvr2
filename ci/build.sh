#!/bin/bash

# Exit script on the first error
set -o errexit -o nounset

mkdir build
cd build
cmake ..
make -j4
