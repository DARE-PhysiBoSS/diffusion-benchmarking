#!/bin/bash

# Define the list of algorithms to test with mpi
algorithms=("MPI_1D_blocking")

# Define the common command parameters
problem_file="example-problems/300x300x300x100.json"
params_file="example-problems/params.json"

# Loop through each algorithm and run the command
for alg in "${algorithms[@]}"; do
    echo "Running benchmark for algorithm: $alg"
    echo "-----------------Single Precison------------------"
    srun build/diffuse_mpi --problem "$problem_file" --params "$params_file" --alg "$alg" --benchmark
    echo "--------------------------------------------------"
    echo "-----------------Double Precison------------------"
    srun build/diffuse_mpi --problem "$problem_file" --params "$params_file" --alg "$alg" --benchmark --double
    echo "--------------------------------------------------"
done