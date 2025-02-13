#!/bin/bash

# Define variables
CPP_FILE="Split22010.cpp"
EXECUTABLE="Split22010"
DATA_DIR="/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/clean_events_dict/22010/22010/reduced"
INPUT_FILE="${DATA_DIR}/clean_event_ids_0000000-0000999.csv"

# Check if the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found!"
    exit 1
fi

# Compile the C++ file
echo "Compiling $CPP_FILE..."
g++ -o "$EXECUTABLE" "$CPP_FILE" -std=c++17
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# Execute the compiled program
echo "Executing $EXECUTABLE..."
./"$EXECUTABLE" "$INPUT_FILE"

# Check if execution was successful
if [ $? -eq 0 ]; then
    echo "Splitting completed! Files saved in: $DATA_DIR"
else
    echo "Error occurred during execution!"
    exit 1
fi
