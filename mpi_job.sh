#!/bin/bash
#SBATCH --ntasks=200               # number of MPI processes
#SBATCH --mem-per-cpu=4G      # memory; default unit is megabytes
#SBATCH --time=0-02:00           # time (DD-HH:MM)

module load scipy-stack mpi4py

srun python simulate_mpi.py
