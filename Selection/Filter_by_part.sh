#!/bin/bash
#SBATCH --job-name=filter_part_%j
#SBATCH --partition=icecube
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyan.jo@proton.me
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Ensure required variables are set
if [[ -z "${SnowstormOrCorsika}" || -z "${SUBDIR}" || -z "${PART}" || -z "${FILTER}" ]]; then
    echo "Error: Required environment variables are missing."
    echo "Usage: sbatch --export=SnowstormOrCorsika=...,SUBDIR=...,PART=...,FILTER=... Filter_by_part.sh"
    exit 1
fi

# Define log directory per filter type
LOG_DIR="log/${SUBDIR}/${FILTER}"

# Ensure the log directory exists
mkdir -p "${LOG_DIR}"

# Generate timestamped log file
timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="${LOG_DIR}/[${timestamp}]log_Filter_${SUBDIR}_${PART}_${FILTER}_${SLURM_JOB_ID}.log"

# Redirect both stdout and stderr to the log file
exec > "${logfile}" 2>&1

echo "Starting job at $(date)"
echo "Running Filter_by_part.py for subdirectory ${SUBDIR}, part ${PART} with filter ${FILTER}"

# Load Python environment if necessary (adjust if using conda/virtualenv)
module load python/3.9  # Modify this if using a specific Python version

# Run the filtering script
python3.9 Filter_by_part.py "${SnowstormOrCorsika}" "${SUBDIR}" "${PART}" "${FILTER}"

echo "Job completed at $(date)"
