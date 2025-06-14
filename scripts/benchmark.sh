#!/bin/bash

# Define the list of algorithms to test
algorithms=("lstmfai" "lstmfabi" "cubedmai" "lstc" "lstcm" "lstcma" "lstct" "lstcta" "lstcs" "lstcst" "lstcsta" "lstcstai" "lstm" "lstmt" "lstmta" "lstmtai" "lstmdt" "lstmdta" "lstmdtai" "lstmdtfa" "lstmdtfai" "avx256d" "biofvm" "lapack" "lapack2" "full_lapack")

# Define the common command parameters
problem_file="example-problems/100x100x100x1.json"
params_file="example-problems/params.json"

# Loop through each algorithm and run the command
for alg in "${algorithms[@]}"; do
    echo "Running benchmark for algorithm: $alg"
    build/diffuse --problem "$problem_file" --params "$params_file" --alg "$alg" --benchmark --double
    echo "-----------------------------------"
done