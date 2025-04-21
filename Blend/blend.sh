#!/bin/bash
#SBATCH --job-name=blending_%j
#SBATCH --partition=icecube
##SBATCH --nodelist=node[195-211] # node[187-191] or node[194-211]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G # 60G or 100G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@60 
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Define log directory per filter combination
LOG_DIR="log/${SUBDIR}/"

# Ensure the log directory exists
mkdir -p "${LOG_DIR}"

# Generate timestamped log file
timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="${LOG_DIR}/[${timestamp}]log_blending_${SLURM_JOB_ID}.log"

# Redirect both stdout and stderr to the log file
exec > "${logfile}" 2>&1

echo "Starting job at $(date)"
echo "Running blend.py"

# Run the filtering script with multiple filters
python3.9 -u blend.py

echo "Job completed at $(date)"


# sbatch --export=SnowstormOrCorsika=Snowstorm,SUBDIR=22011,PART=1 Filter_by_part.sh
# nohup python3.9 Filter_by_part.py Snowstorm 22014 1 > log/22014/[$(date +"%Y%m%d_%H%M%S")]log_Filter_22014_1.log 2>&1 &