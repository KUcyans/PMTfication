#!/bin/bash
#SBATCH --job-name=filter_part_%j
#SBATCH --partition=icecube
##SBATCH --nodelist=node[195-211] # node[187-191] or node[194-211]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G # 60G or 100G
#SBATCH --time=10:00:00
#SBATCH --signal=B:USR1@60 
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Ensure required variables are set
if [[ -z "${SnowstormOrCorsika}" || -z "${SUBDIR}" || -z "${PART}" ]]; then
    echo "Error: Required environment variables are missing."
    echo "Usage: sbatch --export=SnowstormOrCorsika=...,SUBDIR=...,PART=..., Filter_by_part.sh"
    exit 1
fi

# Define log directory per filter combination
LOG_DIR="log/${SUBDIR}/"

# Ensure the log directory exists
mkdir -p "${LOG_DIR}"

# Generate timestamped log file
timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="${LOG_DIR}/[${timestamp}]log_Filter_${SUBDIR}_${PART}_${SLURM_JOB_ID}.log"

# Redirect both stdout and stderr to the log file
exec > "${logfile}" 2>&1

echo "Starting job at $(date)"
echo "Running Filter_by_part.py for subdirectory ${SUBDIR}, part ${PART}"

# Run the filtering script with multiple filters
python3.9 -u Filter_by_part.py "${SnowstormOrCorsika}" "${SUBDIR}" "${PART}"

echo "Job completed at $(date)"


# sbatch --export=SnowstormOrCorsika=Snowstorm,SUBDIR=22011,PART=1 Filter_by_part.sh
# nohup python3.9 Filter_by_part.py Snowstorm 22014 1 > log/22014/[$(date +"%Y%m%d_%H%M%S")]log_Filter_22014_1.log 2>&1 &