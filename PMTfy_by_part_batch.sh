#!/bin/bash

SUBDIR="22010"
SnowstormOrCorsika="Snowstorm"
START_PART=1
END_PART=1
NEVENTS="200000"

for PART in $(seq $START_PART $END_PART); do
    sbatch --export=SnowstormOrCorsika=$SnowstormOrCorsika,SUBDIR=$SUBDIR,PART=$PART,NEVENTS=$NEVENTS PMTfy_by_part.sh
    echo "Submitted job for $SnowstormOrCorsika $SUBDIR part $PART with $NEVENTS events per shard"
done
