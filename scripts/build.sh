#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build

# Navigate to the build directory
cd build

# Run CMake to generate the build files
cmake ..

# Build the project using Make
make -j$(nproc)

# Optionally, install the library to a specific location
# make install

cd ..