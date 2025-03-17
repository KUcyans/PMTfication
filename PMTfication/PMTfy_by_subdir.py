import sys
import os
from os import getenv
import argparse
import logging

from PMTfier import PMTfier

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("PMTfication for a subdirectory starts...")
    
    #### change the final destination: source_root, dest_root
    source_root = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/sqlite_pulses/"
    dest_root = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied/"
    
    source_table_name = 'SRTInIcePulses'
    
    # command-line arguments
    parser = argparse.ArgumentParser(description="PMTfication of a single SQLite database into Parquet files.")
    parser.add_argument("Snowstorm_or_Corsika", type=str, help="Snowstorm or Corsika.")
    parser.add_argument("subdirectory_in_number", type=str, help="Name of the source subdirectory.")
    parser.add_argument("N_events_per_shard", type=int, help="Number of events per shard.")
    args = parser.parse_args()
    
    source_root = os.path.join(source_root, args.Snowstorm_or_Corsika)
    dest_root = os.path.join(dest_root, args.Snowstorm_or_Corsika)
    
    subdirectory_path = os.path.join(source_root, args.subdirectory_in_number)
    try:
        files_count = len(os.listdir(subdirectory_path))
        logging.info(f"The number of files in the subdirectory: {files_count}")
    except FileNotFoundError as e:
        logging.error(f"Directory not found: {e}")
        sys.exit(1)

    max_workers = int(getenv('SLURM_CPUS_PER_TASK', '1'))  # Default to 1 if undefined
    logging.info(f"Using up to {max_workers} workers.")
    
    N_events_per_shard = args.N_events_per_shard
    
    pmtfier = PMTfier(
        source_root= source_root, 
        source_subdirectory=args.subdirectory_in_number,
        source_table=source_table_name,
        dest_root=dest_root,
        N_events_per_shard=N_events_per_shard) 
    pmtfier.pmtfy_subdir_parallel(max_workers=max_workers)
    logging.info("PMTfication completed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)