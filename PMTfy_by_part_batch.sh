#!/bin/bash
# SnowstormOrCorsika: Snowstorm or Corsika
# SUBDIR: Subdirectory of the input files
# PART start : Start part number
# PART end : End part number
# NEVENTS: Number of events per shard

conditions=(
    "Snowstorm 22010 6 11 200000"
    # "Snowstorm 22011 5 10 20000"
    # "Snowstorm 22012 11 21 2000"
    # "Snowstorm 22013 1 5 200000"
    # "Snowstorm 22014 6 6 20000"
    # "Snowstorm 22015 6 12 2000"
    # "Snowstorm 22016 1 5 200000"
    # "Snowstorm 22017 1 5 20000"
    # "Snowstorm 22018 6 21 2000"
    # "Corsika 0002000-0002999 1 8 20000"
    # "Corsika 0003000-0003999 5 9 20000"
    # "Corsika 0004000-0004999 5 9 20000"
    # "Corsika 0005000-0005999 5 9 20000"
    # "Corsika 0006000-0006999 5 9 20000"
    # "Corsika 0007000-0007999 5 9 20000"
    # "Corsika 0008000-0008999 5 8 20000"
    # "Corsika 0009000-0009999 5 9 20000"
)

for condition in "${conditions[@]}"; do
    IFS=' ' read -r SnowstormOrCorsikaValue SUBDIR START_PART END_PART NEVENTS_VALUE <<< "$condition"

    for PART in $(seq $START_PART $END_PART); do
        timestamp=$(date "+%Y-%m-%d %H:%M:%S")
        echo "[$timestamp] Submitting job for $SnowstormOrCorsikaValue $SUBDIR part $PART with $NEVENTS_VALUE events per shard"
        sbatch --export=SnowstormOrCorsika=$SnowstormOrCorsikaValue,SUBDIR=$SUBDIR,PART=$PART,NEVENTS=$NEVENTS_VALUE PMTfy_by_part.sh
    done
done

