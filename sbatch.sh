#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=m4680
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.knyazev@columbia.edu
#SBATCH -J cpu_test

module load conda cray-hdf5-parallel cray-netcdf-hdf5parallel
conda activate simsopt_gpu
srun -n 8 -c 1 python trace_lost_trajectories.py | tee output.txt
