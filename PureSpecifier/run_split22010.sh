#!/bin/bash

CPP_FILE="Split22010.cpp"
EXECUTABLE="Split22010"
DATA_DIR="/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/clean_events_dict/22010/22010/reduced"
MOVE_DIR="/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/clean_events_dict/22010/22010"  # Move target
INPUT_FILE="${DATA_DIR}/clean_event_ids_0000000-0000999.csv"
RENAMED_FILE="reduced_clean_event_ids_0000000-0000999.csv"
OUTPUT_DIR="$DATA_DIR"  # Keep split files in the same directory as input

# Compile the C++ program
g++ -o $EXECUTABLE $CPP_FILE -O2

# Record start time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0

# Run the executable with input file and output directory as arguments
echo "[$START_TIME] Running the script..."
./$EXECUTABLE "$INPUT_FILE" "$OUTPUT_DIR"

# Record end time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$SECONDS

echo "[$END_TIME] Execution Time: $ELAPSED_TIME seconds"

# Rename and move the original CSV file one directory up
if [ -f "$INPUT_FILE" ]; then
    mv "$INPUT_FILE" "${MOVE_DIR}/${RENAMED_FILE}"
    echo "Moved original file to: ${MOVE_DIR}/${RENAMED_FILE}"
else
    echo "Warning: Input file not found, skipping rename and move."
fi
