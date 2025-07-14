#!/bin/bash

sbatch --qos=gp_resa --account=cns119 scripts/run_slurm.sh scripts/profile.sh build counters
sbatch --qos=gp_resa --account=cns119 scripts/run_slurm.sh scripts/benchmark.sh build bench