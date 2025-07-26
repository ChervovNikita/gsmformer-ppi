#!/bin/sh

total_lines=1000
batch_size=1000 
concurrent_jobs=50

echo 'start'

for start in $(seq 1 $batch_size $total_lines); do
    end=$((start + batch_size - 1))

    # Ensure we do not exceed MaxArraySize
    if [ $end -gt 1001 ]; then
        end=1001
    fi

    sbatch --partition=normal  --array=$start-$end%$concurrent_jobs compute_descriptors.slurm
    sleep 0.1
done