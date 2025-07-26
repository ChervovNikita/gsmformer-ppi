#!/bin/bash
#SBATCH --job-name=data_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:00
#SBATCH --array=1-1000%50
#SBATCH --output=logs/data_prepare.%A_%a.out
#SBATCH --error=logs/data_prepare.%A_%a.err

# load your environment here or the Masif docker or aptainer 

SIF_PATH="path_to_sif_file"
SCRIPT_PATH="path_to_script/data_prepare_one.sh" # script from Masif repository 

p=$(sed -n "${SLURM_ARRAY_TASK_ID}p" path_to_list/list.txt) # list of protein complexed in the format pdbID_chain1_chain2 

FIELD1=$(echo $p | cut -d" " -f1)
PDBID=$(echo $FIELD1 | cut -d"_" -f1)
CHAIN1=$(echo $FIELD1 | cut -d"_" -f2)
CHAIN2=$(echo $FIELD1 | cut -d"_" -f3)

singularity exec "$SIF_PATH" "$SCRIPT_PATH" "${PDBID}_${CHAIN1}_${CHAIN2}"

echo "Processed line ${SLURM_ARRAY_TASK_ID}: ${PDBID}_${CHAIN1}_${CHAIN2}"