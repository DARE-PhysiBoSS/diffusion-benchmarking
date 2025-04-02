#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --qos=gp_debug
#SBATCH -t 02:00:00
#SBATCH --account=bsc08
#SBATCH -o output-%j
#SBATCH -e error-%j
##SBATCH --exclusive


export OMP_DISPLAY_ENV=true
export OMP_NUM_THREADS=112
export OMP_PROC_BIND=close
export OMP_PLACES=cores

module purge
module load cmake/3.30.5
module load gcc/13.2.0 openmpi/4.1.5-gcc mkl lapack/3.12-gcc ddt


# Define the list of algorithms to test
algorithms=("lstc" "lstcm" "lstcma" "lstct" "lstcta" "lstcs" "lstcst" "lstcsta" "lstcstai" "lstm" "lstmt" "lstmta" "lstmtai" "avx256d" "biofvm" "lapack" "lapack2" "full_lapack")

# Define the common command parameters
problem_file="example-problems/300x300x300x100.json"
params_file="example-problems/params.json"

# Loop through each algorithm and run the command
for alg in "${algorithms[@]}"; do
    echo "Running benchmark for algorithm: $alg"
    build/diffuse --problem "$problem_file" --params "$params_file" --alg "$alg" --benchmark --double
    echo "-----------------------------------"
done