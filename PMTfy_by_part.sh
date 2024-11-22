#!/bin/bash
#SBATCH --job-name=pmtfication_%j
#SBATCH --partition=icecube
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=96G
#SBATCH --time=5:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyan.jo@proton.me

SUBDIR="22012"
PART="10"
LOG_DIR="log/${SUBDIR}"

mkdir -p "${LOG_DIR}"

timestamp=$(date +"%d%m%Y_%H%M%S")
logfile="${LOG_DIR}/log_PMTfy_${SUBDIR}_${PART}_${SLURM_JOB_ID}_${timestamp}.out"
errfile="${LOG_DIR}/log_PMTfy_${SUBDIR}_${PART}_${SLURM_JOB_ID}_${timestamp}.err"

exec > "${logfile}" 2> "${errfile}"

echo "Starting job at $(date)"
echo "Running PMTfier.py for subdirectory ${SUBDIR} part ${PART} with 2000 events per shard"

python3.9 PMTfy_by_part.py "${SUBDIR}" "${PART}" 2000

echo "Job completed at $(date)"
