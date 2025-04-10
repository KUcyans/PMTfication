#!/bin/bash
# SnowstormOrCorsika: Snowstorm or Corsika
# SUBDIR: Subdirectory of the input files
# PART start : Start part number
# PART end : End part number
# NEVENTS: Number of events per shard
# SUMMARY_MODE: Summary mode index (0=normal, 1=second, 2=late)

SUMMARY_MODE=1  # Change this: 0=normal, 1=second, 2=late

conditions=(
    # "Snowstorm 22010 1 1 200000"         # 1-12
    "Snowstorm 22011 1 10 20000"           # 1-10
    "Snowstorm 22012 1 27 2000"            # 1-27
    # "Snowstorm 22013 1 1 200000"          # 1-2
    "Snowstorm 22014 1 6 20000"            # 1-6
    "Snowstorm 22015 1 12 2000"            # 1-12
    # "Snowstorm 22016 1 1 200000"          # 1-2
    "Snowstorm 22017 1 3 20000"            # 1-3
    "Snowstorm 22018 1 21 2000"            # 1-21
    # "Corsika 0002000-0002999 1 8 20000"   # 1-8
    # "Corsika 0003000-0003999 1 5 20000"   # 1-9
    # "Corsika 0004000-0004999 1 9 20000"   # 1-9
    # "Corsika 0005000-0005999 1 9 20000"   # 1-9
    # "Corsika 0006000-0006999 1 9 20000"   # 1-9
    # "Corsika 0007000-0007999 1 9 20000"   # 1-9
    # "Corsika 0008000-0008999 1 8 20000"   # 1-8
    # "Corsika 0009000-0009999 1 9 20000"   # 1-9

    # "Snowstorm 99999 98 99 10"            # 98, 99
    # "Corsika   9999999-9999999 96 97 10"  # 96, 97
)

for condition in "${conditions[@]}"; do
    IFS=' ' read -r SnowstormOrCorsikaValue SUBDIR START_PART END_PART NEVENTS_VALUE <<< "$condition"

    for PART in $(seq $START_PART $END_PART); do
        timestamp=$(date "+%Y-%m-%d %H:%M:%S")
        echo "[$timestamp] Submitting job for $SnowstormOrCorsikaValue $SUBDIR part $PART with $NEVENTS_VALUE events per shard (SUMMARY_MODE=$SUMMARY_MODE)"
        
        sbatch --export=SnowstormOrCorsika=$SnowstormOrCorsikaValue,SUBDIR=$SUBDIR,PART=$PART,NEVENTS=$NEVENTS_VALUE,SUMMARY_MODE=$SUMMARY_MODE PMTfy_by_part.sh
    done
done
