#!/bin/bash

# Define the list of algorithms to test
algorithms=("ref" "lstc" "lstcm" "lstcma" "lstct" "lstcta" "lstcs" "lstcst" "lstcsta" "lstm" "lstmt" "lstmta" "avx256d" "biofvm" "lapack" "lapack2" "full_lapack")

algorithms_mpi=("MPI_1D_blocking")

# Define the common command parameters
problem_file="example-problems/average.json"

# Loop through each algorithm and run the command
for alg in "${algorithms[@]}"; do
    echo "Running single precision benchmark for algorithm: $alg"
    build/diffuse --problem "$problem_file" --alg "$alg" --validate
    echo "Running double precision benchmark for algorithm: $alg"
    build/diffuse --problem "$problem_file" --alg "$alg" --validate --double
    echo "-----------------------------------"
done