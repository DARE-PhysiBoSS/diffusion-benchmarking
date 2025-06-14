#!/bin/bash

# Define the list of algorithms to test
algorithms=("ref" "lstc" "lstcm" "lstcma" "lstct" "lstcta" "lstcs" "lstcst" "lstcsta" "lstcstai" "lstm" "lstmt" "lstmta" "lstmtai" "lstmdt" "lstmdta" "lstmdtai" "cubed" "biofvm" "avx256d" "lapack" "lapack2" "full_lapack")
algorithms=("full_lapack" )
# Define the common command parameters
problem_files=( "example-problems/toy-small2.json" "example-problems/toy-small.json")
params=("example-problems/params.json" "example-problems/params2.json")

# Loop through each algorithm and run the command
for alg in "${algorithms[@]}"; do
    for param in "${params[@]}"; do 
        for problem_file in "${problem_files[@]}"; do
            echo "Running single precision benchmark with $param parameters and $problem_file problem using algorithm: $alg"
            build/diffuse --problem "$problem_file" --params "$param" --alg "$alg" --validate
            echo "Running double precision benchmark with $param parameters and $problem_file problem using algorithm: $alg"
            build/diffuse --problem "$problem_file" --params "$param" --alg "$alg" --validate --double
        done
    done
    echo "-----------------------------------"
done