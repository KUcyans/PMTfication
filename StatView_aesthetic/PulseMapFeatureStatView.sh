#!/bin/bash
#SBATCH --job-name=pulse_map_feature_stat_view%j
#SBATCH --partition=icecube
##SBATCH --nodelist=node[194-194] # node[187-191] or node[194-211]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

LOG_DIR="log/${SUBDIR}"
mkdir -p "${LOG_DIR}"

timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="${LOG_DIR}/[${timestamp}]FeatureStatView_${SLURM_JOB_ID}.log"

exec > /dev/null 2> "${logfile}"

echo "Starting job at $(date)"

python3.9 -u PulseMapFeatureStatView.py 

echo "Job completed at $(date)"
