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
    logging.info("PMTfication starts...")
    
    source_root = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/sqlite_pulses/"
    dest_root = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied/"
    # dest_root = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_second_round/"

    source_table_name = 'SRTInIcePulses'
    
    snowstorm_source_dir = source_root + "Snowstorm/"
    snowstorm_dest_dir = dest_root + "Snowstorm/"
    corsika_source_dir = source_root + "Corsika/"
    corsika_dest_dir = dest_root + "Corsika/"
    
    parser = argparse.ArgumentParser(description="PMTfication of SQLite databases into Parquet files.")
    parser.add_argument("N_events_per_shard", type=int, help="Number of events per shard.")
    parser.add_argument("--second_round", action="store_true", help="Enable second round of PMTfication.")

    args = parser.parse_args()
    
    snowstorm_sample_subdir = "99999"
    corsika_sample_subdir = "9999999-9999999"
    subdirectory_path_snowstorm = os.path.join(snowstorm_source_dir, snowstorm_sample_subdir)
    subdirectory_path_corsika = os.path.join(corsika_source_dir, corsika_sample_subdir)
    
    try:
        files_count_snowstorm = len(os.listdir(subdirectory_path_snowstorm))
        files_count_corsika = len(os.listdir(subdirectory_path_corsika))
        logging.info(f"The number of files in the subdirectory: {files_count_snowstorm}")
        logging.info(f"The number of files in the subdirectory: {files_count_corsika}")
    except FileNotFoundError as e:
        logging.error(f"Directory not found: {e}")
        sys.exit(1)
    
    max_workers = int(getenv('SLURM_CPUS_PER_TASK', '1'))
    logging.info(f"Using up to {max_workers} workers.")
    
    # PMTfication logic
    N_events_per_shard = args.N_events_per_shard
    
    pmtfier_snowstorm = PMTfier(
        source_root=snowstorm_source_dir,
        source_subdirectory=snowstorm_sample_subdir,
        source_table=source_table_name,
        dest_root=snowstorm_dest_dir,
        N_events_per_shard=N_events_per_shard, 
        is_second_round=args.second_round)
    
    pmtfier_corsika = PMTfier(
        source_root=corsika_source_dir,
        source_subdirectory=corsika_sample_subdir,
        source_table=source_table_name,
        dest_root=corsika_dest_dir,
        N_events_per_shard=N_events_per_shard,
        is_second_round=args.second_round)
    
    logging.info("PMTfying Snowstorm...")
    pmtfier_snowstorm.pmtfy_subdir_parallel(
        max_workers=max_workers)
    logging.info("PMTfication for Snowstorm completed.")
    
    logging.info("PMTfying Corsika...")
    pmtfier_corsika.pmtfy_subdir_parallel(
        max_workers=max_workers)
    logging.info("PMTfication for Corsika completed.")
    
    logging.info("PMTfication completed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)
        
# for the original PMTfication
# nohup python3.9 -u PMTfy_sample.py 10 > log/debug/\[$(date +"%Y%m%d_%H%M%S")\]PMTfy_99999_.log 2>&1 &

# for second round
# nohup python3.9 -u PMTfy_sample.py 10 --second_round > log/debug/[$(date +"%Y%m%d_%H%M%S")]SecondRun_99999_.log 2>&1 &



