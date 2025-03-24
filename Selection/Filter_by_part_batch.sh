#!/bin/bash
# Batch submitter for Filter_by_part.sh
# Iterates over specified dataset conditions and submits SLURM jobs

conditions=(
    # "Snowstorm 22010 10 10"
    "Snowstorm 22011 1 10"
    "Snowstorm 22012 1 26"
    # "Snowstorm 22013 1 2"
    "Snowstorm 22014 1 6"
    "Snowstorm 22015 1 12"
    # "Snowstorm 22016 1 1"
    "Snowstorm 22017 1 3"
    "Snowstorm 22018 1 21"
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
    IFS=' ' read -r SnowstormOrCorsikaValue SUBDIR START_PART END_PART <<< "$condition"
    
    for PART in $(seq $START_PART $END_PART); do
        timestamp=$(date "+%Y-%m-%d %H:%M:%S")
        echo "[$timestamp] Submitting filtering job for $SnowstormOrCorsikaValue $SUBDIR part $PART"
        sbatch --export=SnowstormOrCorsika=$SnowstormOrCorsikaValue,SUBDIR=$SUBDIR,PART=$PART Filter_by_part.sh
    done

done
