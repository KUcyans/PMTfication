#!/bin/bash
#SBATCH --job-name=pmtfication_%j
#SBATCH --partition=icecube
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=8:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyan.jo@proton.me
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

LOG_DIR="log/${SUBDIR}"

mkdir -p "${LOG_DIR}"

timestamp=$(date +"%d%m%Y_%H%M%S")
logfile="${LOG_DIR}/[${timestamp}]log_PMTfy_${SUBDIR}_${PART}_${SLURM_JOB_ID}.out"
errfile="${LOG_DIR}/[${timestamp}]log_PMTfy_${SUBDIR}_${PART}_${SLURM_JOB_ID}.err"

exec > "${logfile}" 2> "${errfile}"

echo "Starting job at $(date)"
echo "Running PMTfier.py for subdirectory ${SUBDIR} part ${PART} with ${NEVENTS} events per shard"

python3.9 PMTfy_by_part.py "${SnowstormOrCorsika}" "${SUBDIR}" "${PART}" "${NEVENTS}"

echo "Job completed at $(date)"
