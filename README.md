# diffusion-benchmarking

This repository contains various variants of parallel implementations of diffusion solver used in PhysiBoSS application. The diffusion application is designed to benchmark these implementation, validate their correctness and, also, solve the provided problems and store the outputed result.

## Build requirements

The requirements to build the application are:
- git
- C++20 compliant compiler
- CMake
- LAPACK library
- OpenMP

> We provide `.devcontainer/devcontainer.json` VSCode Devcontainer, which, after reopening the VSCode window in the Devcontainer, contains all the dependencies to build the application. 

To build `diffuse` app, write:
```bash
# Fetch dependencies
git submodule update --init --recursive
# configure cmake for release build in 'build' directory:
cmake -DCMAKE_BUILD_TYPE=Release -B build
# build the contents of 'build' directory:
cmake --build build
```

## Usage

An example usage of the app looks as follows:
```bash
./build/diffuse --problem example-problems/50x50x50x1.json --alg lstc --params example-problems/params.json --benchmark --double
```
This command benchmarks the diffusion on a problem defined in `example-problems/50x50x50x1.json` file, using algorithm named `lstc` with the parameter set defined in `example-problems/params.json` in double precision. 

Instead of providing `--benchmark` command line parameter, one can use `--validate` to check the correctness of the selected algorithm with respect to the reference algorithm `ref`. Lastly, `--run_and_save FILE` just runs the problem and outputs the result in the provided file.

## List of algorithms

TBA

See `get_solvers_map` function in `src/algorithms.cpp` for the whole list of algorithms and their names.
