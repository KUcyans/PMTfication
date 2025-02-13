import sys
import os
import argparse
import logging

from PureNeutrinoEventFilter import PureNeutrinoEventFilter
from Muon13Filter import Muon13Filter

# Available filter classes
FILTER_CLASSES = {
    # "PureNeutrinoEventFilter": PureNeutrinoEventFilter,
    "Muon13Filter": Muon13Filter,
}

def run():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("Event filtering process starts...")
    
    # Define root paths
    source_root = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied/"
    dest_root = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered/"
    
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Event filtering of PMT-fied data.")
    parser.add_argument("Snowstorm_or_Corsika", type=str, help="Specify 'Snowstorm' or 'Corsika'.")
    parser.add_argument("subdirectory_in_number", type=str, help="Source subdirectory number.")
    parser.add_argument("part_number", type=str, help="Part number.")
    parser.add_argument("filter_types", type=str, help="Comma-separated list of filter class names.")
    args = parser.parse_args()
    
    # Validate and convert numeric arguments
    try:
        subdir_no = int(args.subdirectory_in_number)
        part_no = int(args.part_number)
    except ValueError:
        logging.error("Subdirectory number and part number must be integers.")
        sys.exit(1)

    # Construct paths
    source_root = os.path.join(source_root, args.Snowstorm_or_Corsika)
    dest_root_particular = os.path.join(dest_root, args.Snowstorm_or_Corsika)
    subdirectory_path = os.path.join(source_root, args.subdirectory_in_number)
    source_truth_file = os.path.join(subdirectory_path, f"truth_{args.part_number}.parquet")
    
    # Ensure paths exist
    os.makedirs(dest_root_particular, exist_ok=True)
    if not os.path.isfile(source_truth_file):
        logging.error(f"File not found: {source_truth_file}")
        sys.exit(1)
    if not os.path.isdir(subdirectory_path):
        logging.error(f"Subdirectory not found: {subdirectory_path}")
        sys.exit(1)
    
    # Parse filter types
    selected_filters = args.filter_types.split(",")
    invalid_filters = [f for f in selected_filters if f not in FILTER_CLASSES]
    if invalid_filters:
        logging.error(f"Invalid filter types: {invalid_filters}. Available: {list(FILTER_CLASSES.keys())}")
        sys.exit(1)
    
    # Apply filters in sequence
    input_path = subdirectory_path
    intermediate_output = None
    
    for i, filter_type in enumerate(selected_filters):
        logging.info(f"Applying {filter_type} on subdirectory: {args.subdirectory_in_number}, part: {args.part_number}")

        output_path = os.path.join(dest_root_particular, filter_type)
        os.makedirs(output_path, exist_ok=True)

        source_dir = input_path if i == 0 else intermediate_output
        intermediate_output = output_path  # Set output of this filter for the next iteration

        filter_class = FILTER_CLASSES[filter_type]
        event_filter = filter_class(
            source_dir=source_dir,
            output_dir=output_path,
            subdir_no=subdir_no,
            part_no=part_no
        )
        
        event_filter()
    
    logging.info("Filtering process completed for all selected filters.")

def main():
    run()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)
