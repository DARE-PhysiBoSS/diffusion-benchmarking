#!/bin/bash

# Define the list of algorithms to test
algorithms=("ref" "lstc" "lstcm" "lstcma" "lstct" "lstcta" "lstcs" "lstcst" "lstcsta" "lstm" "lstmt" "lstmta" "avx256d" "biofvm" "lapack" "lapack2" "full_lapack")

# Define the common command parameters
problem_file="example-problems/toy-small.json"

# Loop through each algorithm and run the command
for alg in "${algorithms[@]}"; do
    echo "Running benchmark for algorithm: $alg"
    build/diffuse --problem "$problem_file" --alg "$alg" --validate
    echo "-----------------------------------"
done