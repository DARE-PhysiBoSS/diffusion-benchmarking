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
export OMP_PLACES=threads

module purge
module load cmake/3.30.5
module load gcc/13.2.0 openmpi/4.1.5-gcc mkl lapack/3.12-gcc ddt


./build/diffuse --alg ref --problem ./example-problems/50x50x50x1.json --benchmark --double >> test1.csv
./build/diffuse --alg lstc --problem ./example-problems/50x50x50x1.json --benchmark --double >> test1.csv
./build/diffuse --alg lstc_t --problem ./example-problems/50x50x50x1.json --benchmark --double >> test1.csv
./build/diffuse --alg lstm --problem ./example-problems/50x50x50x1.json --benchmark --double >> test1.csv
./build/diffuse --alg lapack --problem ./example-problems/50x50x50x1.json --benchmark --double >> test1.csv
./build/diffuse --alg lapack2 --problem ./example-problems/50x50x50x1.json --benchmark --double >> test1.csv
./build/diffuse --alg full_lapack --problem ./example-problems/50x50x50x1.json --benchmark --double >> test1.csv
./build/diffuse --alg avx256d --problem ./example-problems/50x50x50x1.json --benchmark --double >> test1.csv

./build/diffuse --alg ref --problem ./example-problems/50x50x50x8.json --benchmark --double >> test2.csv
./build/diffuse --alg lstc --problem ./example-problems/50x50x50x8.json --benchmark --double >> test2.csv
./build/diffuse --alg lstc_t --problem ./example-problems/50x50x50x8.json --benchmark --double >> test2.csv
./build/diffuse --alg lstm --problem ./example-problems/50x50x50x8.json --benchmark --double >> test2.csv
./build/diffuse --alg lapack --problem ./example-problems/50x50x50x8.json --benchmark --double >> test2.csv
./build/diffuse --alg lapack2 --problem ./example-problems/50x50x50x8.json --benchmark --double >> test2.csv
./build/diffuse --alg full_lapack --problem ./example-problems/50x50x50x8.json --benchmark --double >> test2.csv
./build/diffuse --alg avx256d --problem ./example-problems/50x50x50x8.json --benchmark --double >> test2.csv