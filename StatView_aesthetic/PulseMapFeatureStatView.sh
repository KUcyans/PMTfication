#!/bin/bash
#SBATCH --job-name=pulse_map_feature_stat_view%j
#SBATCH --partition=icecube
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

ENERGY_ID=${ENERGY_ID:-0}   # 0 = 10TeV–1PeV, 1 = 1PeV–100PeV, 2 = 100TeV–100PeV
FLAVOUR_ID=${FLAVOUR_ID:-0} # 0 = nu_e, 1 = nu_mu, 2 = nu_tau
N_EVENTS=${N_EVENTS:-10000} # Number of events to process

ENERGY_ID=1
FLAVOUR_ID=2 
N_EVENTS=10000

LOG_DIR="log/"
mkdir -p "${LOG_DIR}"

timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="${LOG_DIR}/[${timestamp}]FeatureStatView_er${ENERGY_ID}_fl${FLAVOUR_ID}.log"

exec > "${logfile}" 2>&1

echo "Starting job at $(date)"
echo "Energy ID: ${ENERGY_ID}, Flavour ID: ${FLAVOUR_ID}"

python3.9 -u PulseMapFeatureStatView.py "${N_EVENTS}" "${ENERGY_ID}" "${FLAVOUR_ID}"

echo "Job completed at $(date)"
