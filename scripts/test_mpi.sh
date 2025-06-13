#!/bin/sh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --qos=gp_debug
#SBATCH -t 02:00:00
#SBATCH --account=bsc08
#SBATCH -o output-%j
#SBATCH -e error-%j
#SBATCH --exclusive
#SBATCH --constraint=perfparanoid 

#Validate mpi algorithms

algorithms_mpi=("MPI_1D_blocking")

NUM_CORES=$(nproc)

export OMP_NUM_THREADS=$NUM_CORES
export OMP_PLACES=cores
export OMP_PROC_BIND=close

problem_file="example-problems/average.json"
params_file="example-problems/params.json"



for alg in "${algorithms_mpi[@]}"; do
    echo "Running single precision benchmark for algorithm: $alg"
    #mpirun --map-by ppr:1:node build/diffuse_mpi --problem "$problem_file" --alg "$alg" --validate
    srun --nodes=4 --ntasks-per-node=1 --cpus-per-task=112 build/diffuse_mpi --problem "$problem_file" --params "$params_file" --alg "$alg" --validate
    echo "Running double precision benchmark for algorithm: $alg"
    #mpirun --map-by ppr:1:node build/diffuse_mpi --problem "$problem_file" --alg "$alg" --validate --double
    srun --nodes=4 --ntasks-per-node=1 --cpus-per-task=112 build/diffuse_mpi --problem "$problem_file" --params "$params_file" --alg "$alg" --validate --double
    echo "-----------------------------------"
done
