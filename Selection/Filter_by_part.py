import sys
import os
import argparse
import logging
import time

from PureNeutrinoEventFilter import PureNeutrinoEventFilter
from Muon13Filter import Muon13Filter

# Available filter classes
FILTER_CLASSES = {
    "PureNu": PureNeutrinoEventFilter,
    # "MuonLike": Muon13Filter,s
}

def run():
    start_time = time.time()
    
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
    # dest_root_particular = os.path.join(dest_root, args.Snowstorm_or_Corsika) #PMTfied_filtered/Snowstorm/
    subdirectory_path = os.path.join(source_root, args.subdirectory_in_number)
    source_truth_file = os.path.join(subdirectory_path, f"truth_{args.part_number}.parquet")
    
    # os.makedirs(dest_root_particular, exist_ok=True)
    if not os.path.isfile(source_truth_file):
        logging.error(f"File not found: {source_truth_file}")
        sys.exit(1)
    if not os.path.isdir(subdirectory_path):
        logging.error(f"Subdirectory not found: {subdirectory_path}")
        sys.exit(1)
    
    input_path = subdirectory_path
    
    for i, (filter_key, filter_class) in enumerate(FILTER_CLASSES.items()):
        dest_dir = os.path.join(dest_root, filter_key, args.Snowstorm_or_Corsika, args.subdirectory_in_number)
        os.makedirs(dest_dir, exist_ok=True)

        event_filter = filter_class(
            source_dir=input_path,
            output_dir=dest_dir,
            subdir_no=subdir_no,
            part_no=part_no
        )
        
        event_filter()
    
    logging.info("Filtering process completed for all selected filters.")
    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"Execution time: {end_time - start_time:.2f} seconds")
    event_filter.update_receipt_time(start_time=start_time, end_time=end_time, duration=duration)

def main():
    run()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)
