#!/bin/bash
#SBATCH --job-name=pmtfication_%j
#SBATCH --partition=icecube
##SBATCH --nodelist=node[194-194] # node[187-191] or node[194-211]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyan.jo@proton.me
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Summary mode index: 0=normal, 1=second, 2=late
SUMMARY_MODE=${SUMMARY_MODE:-0}  # Default to 0 if not set

LOG_DIR="log/${SUBDIR}"
mkdir -p "${LOG_DIR}"

timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="${LOG_DIR}/[${timestamp}]PMTfy_${SUBDIR}_${PART}_${SLURM_JOB_ID}.log"

exec > /dev/null 2> "${logfile}"

echo "Starting job at $(date)"
echo "Running PMTfier.py for subdirectory ${SUBDIR} part ${PART} with ${NEVENTS} events per shard"
echo "Summary mode index: ${SUMMARY_MODE}"

python3.9 -u PMTfy_by_part.py "${SnowstormOrCorsika}" "${SUBDIR}" "${PART}" "${NEVENTS}" --summary_mode "${SUMMARY_MODE}"

echo "Job completed at $(date)"

# EMAIL_SUBJECT="SLURM Job Notification: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
# EMAIL_BODY="Job Details:\n\
# - Snowstorm or Corsika: ${SnowstormOrCorsika}\n\
# - Subdirectory: ${SUBDIR}\n\
# - Part: ${PART}\n\
# - Events per shard: ${NEVENTS}\n\
# - Summary Mode: ${SUMMARY_MODE}\n\
# - Job Name: ${SLURM_JOB_NAME}\n\
# - Job ID: ${SLURM_JOB_ID}\n\
# - Partition: ${SLURM_JOB_PARTITION}\n\
# - Completed At: $(date)\n\n\
# Log Snippet:\n$(tail -n 20 "${logfile}")"

# echo -e "${EMAIL_BODY}" | mailx -s "${EMAIL_SUBJECT}" cyan.jo@proton.me
