#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=56
#SBATCH --qos=gp_debug
#SBATCH -t 02:00:00
#SBATCH --account=bsc08
#SBATCH -o output-%j
#SBATCH -e error-%j
#SBATCH --exclusive


export OMP_DISPLAY_ENV=true
#export OMP_NUM_THREADS=48
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

module purge
module load cmake/3.30.5
module load gcc/13.2.0 openmpi/4.1.5-gcc mkl lapack/3.12-gcc 