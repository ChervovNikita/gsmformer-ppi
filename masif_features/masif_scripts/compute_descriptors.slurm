#!/bin/bash
#SBATCH --job-name=task_name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00-00:00
#SBATCH --array=1-1000%50
#SBATCH --output=task_name_.%A_%a.out
#SBATCH --error=task_name_.%A_%a.err

# load environment 

export IBV_FORK_SAFE=1
export FI_PROVIDER=tcp


SIF_PATH="path_to_sif"
SCRIPT_PATH="/compute_descriptors.sh"

p=$(sed -n "${SLURM_ARRAY_TASK_ID}p" lists/test2_masif.txt)

FIELD1=$(echo $p | cut -d" " -f1)
PDBID=$(echo $FIELD1 | cut -d"_" -f1)
CHAIN1=$(echo $FIELD1 | cut -d"_" -f2)
CHAIN2=$(echo $FIELD1 | cut -d"_" -f3)

singularity exec "$SIF_PATH" "$SCRIPT_PATH" "${PDBID}_${CHAIN1}_${CHAIN2}"

echo "Processed line ${SLURM_ARRAY_TASK_ID}: ${PDBID}_${CHAIN1}_${CHAIN2}"
