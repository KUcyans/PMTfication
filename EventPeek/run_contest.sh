#!/bin/bash
#SBATCH --job-name=event_contest_%j
#SBATCH --partition=icecube
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --exclusive
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

LOG_DIR="log/"

mkdir -p "${LOG_DIR}"

timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="${LOG_DIR}/[${timestamp}]FeatureChampionship.log"

# Redirect stdout and stderr to logfile
exec > "${logfile}" 2>&1

echo "Starting job at $(date)"
echo "Let the show begin"

python3.9 -u NewFeatureChampionship.py

echo "Job completed at $(date)"
echo "The show is over"
