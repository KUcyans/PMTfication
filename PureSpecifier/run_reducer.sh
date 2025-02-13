#!/bin/bash

# Define parameters
CPP_FILE="PureNuSpecifierReducer.cpp"
EXECUTABLE="purifier"
BASE_DIR="/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/clean_events_dict"

# Compile the C++ program with optimizations
g++ -O2 -o "$EXECUTABLE" "$CPP_FILE"

# Check if compilation succeeded
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# Iterate over subdirectories 22010 to 22018
for SUBDIR in {22010..22018}; do
    INPUT_DIR="${BASE_DIR}/${SUBDIR}/${SUBDIR}"
    OUTPUT_DIR="${INPUT_DIR}/reduced"

    # Check if input directory exists
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Skipping $INPUT_DIR (directory not found)"
        continue
    fi

    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"

    echo "Processing files in $SUBDIR"

    # Process all CSV files in the directory
    for INPUT_FILE in "$INPUT_DIR"/*.csv; do
        # Extract filename without directory path
        FILE_NAME=$(basename "$INPUT_FILE")

        # Define output file path
        OUTPUT_FILE="$OUTPUT_DIR/$FILE_NAME"

        echo "Processing: $INPUT_FILE -> $OUTPUT_FILE"

        # Run the C++ program
        ./"$EXECUTABLE" "$INPUT_FILE" "$OUTPUT_FILE"

        # Check if execution succeeded
        if [ $? -ne 0 ]; then
            echo "Error processing $INPUT_FILE"
            exit 1
        fi
    done
done

echo "All files processed successfully."
