#!/bin/bash

# Define the list of algorithms to test
algorithms=("lstc" "lstcm" "lstcma" "lstct" "lstcta" "lstcs" "lstcst" "lstcsta" "lstm" "lstmt" "lstmta" "avx256d" "biofvm" "lapack" "lapack2" "full_lapack")

# Define the common command parameters
problem_file="example-problems/300x300x300x100.json"
params_file="example-problems/params.json"

# Loop through each algorithm and run the command
for alg in "${algorithms[@]}"; do
    echo "Running benchmark for algorithm: $alg"
    build/diffuse --problem "$problem_file" --params "$params_file" --alg "$alg" --benchmark
    echo "-----------------------------------"
done