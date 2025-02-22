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
    source_table_name = 'SRTInIcePulses'
    
    snowstorm_source_dir = source_root + "Snowstorm/"
    snowstorm_dest_dir = dest_root + "Snowstorm/"
    corsika_source_dir = source_root + "Corsika/"
    corsika_dest_dir = dest_root + "Corsika/"
    
    parser = argparse.ArgumentParser(description="PMTfication of SQLite databases into Parquet files.")
    parser.add_argument("N_events_per_shard", type=int, help="Number of events per shard.")
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
        N_events_per_shard=N_events_per_shard)
    
    pmtfier_corsika = PMTfier(
        source_root=corsika_source_dir,
        source_subdirectory=corsika_sample_subdir,
        source_table=source_table_name,
        dest_root=corsika_dest_dir,
        N_events_per_shard=N_events_per_shard)
    
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
        
    
# nohup python3.9 PMTfy_sample.py 10 > log/debug/\[$(date +"%d%m%Y_%H%M%S")\]_PMTfy_99999_.log 2>&1 &
