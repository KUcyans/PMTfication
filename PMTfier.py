import sqlite3 as sql
import pandas as pd
import sys
import os
import numpy as np
from os import getenv

from typing import List

import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm
import argparse

import logging

from concurrent.futures import as_completed, ProcessPoolExecutor

from multiprocessing import cpu_count

from PMT_summariser import PMTSummariser
from PMT_truth_maker import PMTTruthMaker

class PMTfier:
    def __init__(self, 
                source_table: str, 
                dest_root: str, 
                N_events_per_shard: int):
        self.source_table = source_table
        self.dest_root = dest_root
        self.N_events_per_shard = N_events_per_shard
        
    def _get_table_event_count(self, conn: sql.Connection, table: str) -> int:
        cursor = conn.cursor()
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_event_no ON {table}(event_no)")
        cursor.execute(f"SELECT COUNT(DISTINCT event_no) FROM {table}")
        event_count = cursor.fetchone()[0]
        return event_count
    
    def _get_event_no_batches(self, conn: sql.Connection, table: str, batch_size: int):
        last_event_no = None
        while True:
            query = f"""
            SELECT DISTINCT event_no
            FROM {table}
            {f"WHERE event_no > {last_event_no}" if last_event_no else ""}
            ORDER BY event_no
            LIMIT {batch_size}
            """
            batch = pd.read_sql_query(query, conn)['event_no'].tolist()
            if not batch:
                break
            last_event_no = batch[-1]
            yield batch

    def _get_truth_table_name_db(self, con_source: sql.Connection) -> str:
        cur_source = con_source.cursor()
        cur_source.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur_source.fetchall()]
        if 'truth' in tables:
            truth_table = 'truth'
        elif 'Truth' in tables:
            truth_table = 'Truth'
        else:
            raise ValueError("Neither 'truth' nor 'Truth' table exists in the source database.")
        return truth_table
    
    def _get_subdir_tag(self, subdir: str) -> int:
        strSnowstromOrCorsika = os.path.basename(os.path.normpath(self.dest_root))
        if strSnowstromOrCorsika == 'Snowstorm':
            # "22012" -> 12
            return int(subdir[-2:])
        elif strSnowstromOrCorsika == 'Corsika':
            # "0002000-0002999" -> 20
            # "0003000-0003999" -> 30
            range_start = subdir.split('-')[0]  # "0002000"
            range_end = subdir.split('-')[1]  # "0002999"
            
            # Extracting digits from the respective ranges
            first_half_subdir_no = range_start[3:5]  # "20"
            second_half_subdir_no = range_end[3:5]  # "29"
            
            # Combine the two parts into a single integer
            combined_subdir_no = int(first_half_subdir_no + second_half_subdir_no)
            return combined_subdir_no
            
    def _get_part_no(self, file: str) -> int:
        """
        file (str): get the number of the file (e.g., "merged_part_1.db").
        """
        try:
            file_no = int(file.split('_')[-1].split('.')[0])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid file name format: '{file}' does not contain a numeric identifier.")
        
        return file_no
    
    def _add_enhance_event_no(self, pa_table: pa.Table, subdir_tag: int, part_no: int) -> pa.Table:
        """
        event_no transformed to a unique identifier for the entire dataset.
        """
        if 'event_no' in pa_table.schema.names:
            original_event_no = pa_table['event_no']
            enhanced_event_no = [
                int(f"{subdir_tag:02}{part_no:04}{event_no.as_py():08}")
                for event_no in original_event_no]

            pa_table = pa_table.remove_column(pa_table.schema.get_field_index('event_no'))
            pa_table = pa_table.append_column('original_event_no', original_event_no)
            pa_table = pa_table.append_column('event_no', pa.array(enhanced_event_no))

            reordered_columns = ['event_no', 'original_event_no'] + [
                name for name in pa_table.schema.names if name not in ('event_no', 'original_event_no')
            ]
            pa_table = pa_table.select(reordered_columns)

        return pa_table
        
    def pmtfy_shard(self, 
                    con_source: sql.Connection,
                    source_table: str,
                    truth_table_name: str,
                    source_subdirectory: str,
                    part_no: int,
                    shard_no: int,
                    event_batch: List[int]) -> pd.DataFrame:
        """
        Processes a shard of events from the database, saves PMTfied data to a file, and returns the truth+receipt data.
        """
        # Query the subset of event_no for this shard
        # event_no_query = f"""
        #     SELECT DISTINCT event_no 
        #     FROM {source_table}
        #     ORDER BY event_no ASC
        #     LIMIT {limit} OFFSET {offset}
        # """
        # event_no_subset = pd.read_sql_query(event_no_query, con_source)['event_no'].tolist()
        subdir_tag = self._get_subdir_tag(source_subdirectory)
        dest_dir = os.path.join(self.dest_root, source_subdirectory, str(part_no))
        os.makedirs(dest_dir, exist_ok=True)
        
        # NOTE
        # PMTSummariser is the core class to be called for PMTfication
        summariser  = PMTSummariser(
            con_source=con_source,
            source_table=source_table,
            event_no_subset=event_batch
        pa_pmtfied = summariser()
        pa_pmtfied = self._add_enhance_event_no(pa_pmtfied, subdir_tag, part_no)
        pmtfied_file = os.path.join(dest_dir, f"PMTfied_{shard_no}.parquet")        
        pq.write_table(pa_pmtfied, pmtfied_file)
        
        # NOTE
        # PMT truth table for this shard is created by PMTTruthMaker and returned
        veritator = PMTTruthMaker(con_source, source_table, truth_table_name, event_no_subset)
        pa_truth_shard = veritator(part_no, shard_no, int(source_subdirectory))
        pa_truth_shard = self._add_enhance_event_no(pa_truth_shard, subdir_tag, part_no)
        
        return pa_truth_shard
        
    def _divide_and_conquer_part(self, 
                            con_source: sql.Connection, 
                            source_table: str, 
                            truth_table_name: str,
                            source_subdirectory: str, 
                            part_no: int, 
                            N_events_per_shard: int) -> pd.DataFrame:
        """
        Divides the database events into shards and processes each shard, consolidating the truth+receipt data.
        """
        truth_shards = []
        
        for shard_no, event_batch in enumerate(self._get_event_no_batches(con_source, source_table, N_events_per_shard), start=1):
            pa_truth_shard = self.pmtfy_shard(
                con_source=con_source,
                source_table=source_table,
                truth_table_name=truth_table_name,
                source_subdirectory=source_subdirectory,
                part_no=part_no,
                shard_no=shard_no + 1)
            truth_shards.append(pa_truth_shard)

        consolidated_truth = pa.concat_tables(truth_shards)
        return consolidated_truth
        
    def pmtfy_part(self,
                source_subdirectory: str,
                source_file: str,
                source_table: str,
                N_events_per_shard: int = 2000) -> None:
        part_no = self._get_part_no(source_file)
        con_source = sql.connect(source_file)
        truth_table_name = self._get_truth_table_name_db(con_source)
        
        consolidated_truth = self._divide_and_conquer_part(
            con_source=con_source,
            source_table=source_table,
            truth_table_name=truth_table_name,
            source_subdirectory=source_subdirectory,
            part_no=part_no,
            N_events_per_shard=N_events_per_shard)

        dest_subdirectory_path = os.path.join(self.dest_root, source_subdirectory)
        os.makedirs(dest_subdirectory_path, exist_ok=True)
        consolidated_file = os.path.join(dest_subdirectory_path, f"truth_{part_no}.parquet")
        pq.write_table(consolidated_truth, consolidated_file)

        con_source.close()
    
    def pmtfy_subdir_parallel(self, 
                            source_root: str, 
                            subdirectory_name: str, 
                            source_table: str, 
                            N_events_per_shard: int = 2000, 
                            max_workers: int = 5) -> None:
        """
        Processes each database file in a specific subdirectory using parallel workers 
        and saves results in a mirrored directory structure.
        """
        subdirectory_path = os.path.join(source_root, subdirectory_name)
        if os.path.isdir(subdirectory_path) and subdirectory_name.isdigit():
            files = [f for f in os.listdir(subdirectory_path) if f.endswith('.db') and os.path.isfile(os.path.join(subdirectory_path, f))]
            logging.info(f"Found {len(files)} database files in subdirectory {subdirectory_name}.")
            if max_workers is None:
                max_workers = min(len(files), cpu_count() // 2)
            # Adjust max_workers based on the number of files
            # actual_workers = min(len(files), max_workers)
            # logging.info(f"Using {actual_workers} workers for parallel processing.")
            
            # Process files in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                executor.submit(self.pmtfy_part, 
                                source_subdirectory=subdirectory_name,
                                source_file=os.path.join(subdirectory_path, filename),
                                source_table=source_table,
                                N_events_per_shard=N_events_per_shard): filename
                for filename in files
            }
                for future in tqdm(as_completed(futures), desc=f"Processing {subdirectory_name}", total=len(files)):
                    filename = futures[future]
                    try:
                        future.result()
                        logging.info(f"Finished processing database file: {filename}")
                    except Exception as e:
                        logging.error(f"Error processing file {filename}: {str(e)}", exc_info=True)
                    
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("PMTfication starts...")
    
    # User-defined constants
    source_root = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/sqlite_pulses/Snowstorm/"
    dest_root = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied/Snowstorm/"
    source_table_name = 'SRTInIcePulses'
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="PMTfication of SQLite databases into Parquet files.")
    parser.add_argument("subdirectory_in_number", type=str, help="Name of the source subdirectory.")
    parser.add_argument("N_events_per_shard", type=int, help="Number of events per shard.")
    args = parser.parse_args()
    
    # Logging the subdirectory being processed
    subdirectory_path = os.path.join(source_root, args.subdirectory_in_number)
    try:
        files_count = len(os.listdir(subdirectory_path))
        logging.info(f"The number of files in the subdirectory: {files_count}")
    except FileNotFoundError as e:
        logging.error(f"Directory not found: {e}")
        sys.exit(1)
    
    max_workers = int(getenv('SLURM_CPUS_PER_TASK', '1'))  # Default to 1 if undefined
    logging.info(f"Using up to {max_workers} workers.")
    
    # PMTfication logic
    N_events_per_shard = args.N_events_per_shard
    pmtfier = PMTfier(source_table_name, dest_root, N_events_per_shard)
    pmtfier.pmtfy_subdir_parallel(
        source_root=source_root,
        subdirectory_name=args.subdirectory_in_number,
        source_table=source_table_name,
        N_events_per_shard=N_events_per_shard,
        max_workers=max_workers
    )
    logging.info("PMTfication completed.")

# Wrap the main function in a try-except block
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)