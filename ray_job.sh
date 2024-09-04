#!/bin/bash
#SBATCH --nodes 2
#SBATCH --cpus-per-task=8
#SBATCH --time=0-00:10

## Create a virtualenv and install Ray on all nodes ##
srun -N $SLURM_NNODES -n $SLURM_NNODES config_env.sh

export HEAD_NODE=$(hostname) # store head node's address
export RAY_PORT=34567 # choose a port to start Ray on the head node 

source $SLURM_TMPDIR/ENV/bin/activate

## Start Ray cluster Head Node ##
ray start --head --node-ip-address=$HEAD_NODE --port=$RAY_PORT --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=1 --block &
sleep 10

## Launch worker nodes on all the other nodes allocated by the job ##
srun launch_ray.sh &
ray_cluster_pid=$!

python test_ray.py

## Shut down Ray worker nodes after the Python script exits ##
kill $ray_cluster_pid
