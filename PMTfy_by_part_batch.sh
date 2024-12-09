#!/bin/bash
# SnowstormOrCorsika: Snowstorm or Corsika
# SUBDIR: Subdirectory of the input files
# PART start : Start part number
# PART end : End part number
# NEVENTS: Number of events per shard

conditions=(
    "Snowstorm 22010 3 3 200000"
    "Snowstorm 22011 3 3 20000"
    "Snowstorm 22012 3 3 2000"
    "Corsika 0003000-0003999 3 3 20000"
    "Corsika 0004000-0004999 3 3 20000"
    "Corsika 0005000-0005999 3 3 20000"
)

for condition in "${conditions[@]}"; do
    IFS=' ' read -r SnowstormOrCorsikaValue SUBDIR START_PART END_PART NEVENTS_VALUE <<< "$condition"

    for PART in $(seq $START_PART $END_PART); do
        timestamp=$(date "+%Y-%m-%d %H:%M:%S")
        echo "[$timestamp] Submitting job for $SnowstormOrCorsikaValue $SUBDIR part $PART with $NEVENTS_VALUE events per shard"
        sbatch --export=SnowstormOrCorsika=$SnowstormOrCorsikaValue,SUBDIR=$SUBDIR,PART=$PART,NEVENTS=$NEVENTS_VALUE PMTfy_by_part.sh
    done
done

