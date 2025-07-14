#!/bin/bash

sbatch --qos=gp_hbm --account=cns119 scripts/run_slurm.sh scripts/profile.sh build-hbm counters-hbm
sbatch --qos=gp_hbm --account=cns119 scripts/run_slurm.sh scripts/benchmark.sh build-hbm bench-hbm
