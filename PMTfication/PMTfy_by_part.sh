#!/bin/bash
#SBATCH --job-name=pmtfication_%j
#SBATCH --partition=icecube
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyan.jo@proton.me
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

LOG_DIR="log/${SUBDIR}"
mkdir -p "${LOG_DIR}"

timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="${LOG_DIR}/[${timestamp}]PMTfy_${SUBDIR}_${PART}_${SLURM_JOB_ID}.log"

exec > /dev/null 2> "${logfile}"

echo "Starting job at $(date)"
echo "Running PMTfier.py for subdirectory ${SUBDIR} part ${PART} with ${NEVENTS} events per shard"

# Add --second_round if enabled
SECOND_ROUND_FLAG=""
if [[ "${SECOND_ROUND}" == "true" ]]; then
    SECOND_ROUND_FLAG="--second_round"
    echo "Running in SECOND ROUND mode."
fi

python3.9 -u PMTfy_by_part.py "${SnowstormOrCorsika}" "${SUBDIR}" "${PART}" "${NEVENTS}" ${SECOND_ROUND_FLAG}

echo "Job completed at $(date)"

EMAIL_SUBJECT="SLURM Job Notification: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
EMAIL_BODY="Job Details:\n\
- Snowstorm or Corsika: ${SnowstormOrCorsika}\n\
- Subdirectory: ${SUBDIR}\n\
- Part: ${PART}\n\
- Events per shard: ${NEVENTS}\n\
- Job Name: ${SLURM_JOB_NAME}\n\
- Job ID: ${SLURM_JOB_ID}\n\
- Partition: ${SLURM_JOB_PARTITION}\n\
- Second Round: ${SECOND_ROUND}\n\
- Completed At: $(date)\n\n\
Log Snippet:\n$(tail -n 20 "${logfile}")"

echo -e "${EMAIL_BODY}" | mailx -s "${EMAIL_SUBJECT}" cyan.jo@proton.me
