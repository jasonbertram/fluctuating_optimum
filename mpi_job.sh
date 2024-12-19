#!/bin/bash
#SBATCH --ntasks=100               # number of MPI processes
#SBATCH --mem-per-cpu=2G      # memory; default unit is megabytes
#SBATCH --time=0-10:00           # time (DD-HH:MM)

module load scipy-stack mpi4py

srun python simulate_mpi.py
