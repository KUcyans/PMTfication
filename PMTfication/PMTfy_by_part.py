import sys
import os
from os import getenv
import argparse
import logging
from PMTfier import PMTfier
from SummaryMode import SummaryMode

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("PMTfication by part starts...")
    
    #### change the final destination: source_root, dest_root
    source_root = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/sqlite_pulses/"
    dest_root_base = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied"
    
    source_table_name = 'SRTInIcePulses'
    
    # Command-line arguments
    parser = argparse.ArgumentParser(description="PMTfication of a single SQLite database into Parquet files.")
    parser.add_argument("Snowstorm_or_Corsika", type=str, help="Snowstorm or Corsika.")
    parser.add_argument("subdirectory_in_number", type=str, help="Name of the source subdirectory.")
    parser.add_argument("part_number", type=str, help="Part number.")
    parser.add_argument("N_events_per_shard", type=int, help="Number of events per shard.")
    parser.add_argument("--summary_mode", type=int, choices=[0, 1, 2], default=0, help="Summary mode: 0=normal, 1=second, 2=late (default: 0)")

    args = parser.parse_args()

    # Update source and destination directories
    source_root = os.path.join(source_root, args.Snowstorm_or_Corsika)
    summary_mode = SummaryMode.from_index(args.summary_mode)
    suffix = "" if summary_mode == SummaryMode.CLASSIC else f"_{summary_mode}"
    dest_root = os.path.join(dest_root_base + suffix, args.Snowstorm_or_Corsika)

    subdirectory_path = os.path.join(source_root, args.subdirectory_in_number)
    source_file_path = os.path.join(subdirectory_path, f"merged_part_{args.part_number}.db")
    
    if not os.path.isfile(source_file_path):
        logging.error(f"File not found: {source_file_path}")
        sys.exit(1)
    
    source_file = os.path.basename(source_file_path)
    logging.info(f"Processing file: {source_file}")
    
    max_workers = int(getenv('SLURM_CPUS_PER_TASK', '1'))
    logging.info(f"Using up to {max_workers} workers.")
    
    N_events_per_shard = args.N_events_per_shard
    pmtfier = PMTfier(
        source_root=source_root, 
        source_subdirectory=args.subdirectory_in_number,
        source_table=source_table_name,
        dest_root=dest_root,
        N_events_per_shard=N_events_per_shard,
        summary_mode=summary_mode,
    ) 
    
    pmtfier.pmtfy_part(source_part_file=source_file_path)
    logging.info("PMTfication completed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)
