#!/bin/bash

source $SLURM_TMPDIR/ENV/bin/activate
module load gcc/9.3.0 arrow

if [[ "$SLURM_PROCID" -eq "0" ]]; then
        echo "Ray head node already started..."
        sleep 10

else
        ray start --address "${HEAD_NODE}:${RAY_PORT}" --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus=1 --block
        sleep 5
        echo "ray worker started!"
fi