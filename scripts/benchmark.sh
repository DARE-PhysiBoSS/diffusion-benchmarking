#!/bin/bash

# Define the list of algorithms to test
algorithms=("lstc" "lstcm" "lstcma" "lstct" "lstcta" "lstcs" "lstcst" "lstcsta" "lstcstai" "lstm" "lstmt" "lstmta" "lstmtai" "avx256d" "biofvm" "lapack" "lapack2" "full_lapack" "cr" "crt" "sblocked" "blocked" "blockedt" "blockedta" "cubed")

# Define the common command parameters
problem_file="example-problems/liver/10%_liver.json"
params_file="example-problems/params.json"

# Loop through each algorithm and run the command
for alg in "${algorithms[@]}"; do
    echo "Running benchmark for algorithm: $alg"
    build/diffuse --problem "$problem_file" --params "$params_file" --alg "$alg" --benchmark
    echo "-----------------------------------"
done