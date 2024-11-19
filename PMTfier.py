import sqlite3 as sql
import pandas as pd
import sys
import os
import numpy as np

from typing import List

import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm
import argparse

import logging

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

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
        cursor.execute(f"SELECT COUNT(DISTINCT event_no) FROM {table}")
        event_count = cursor.fetchone()[0]
        return event_count
    
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
    
    def _generate_truth_for_part(self, con_source, source_table, truth_table_name, event_no_subsets, part_no):
        """
        Generate truth data for the entire part, then split it into shards.

        Args:
            con_source: SQLite connection to the database.
            source_table: Name of the source table.
            truth_table_name: Name of the truth table.
            event_no_subsets: List of event_no lists for each shard.
            part_no: Part number for the current database file.

        Returns:
            List of PyArrow Tables, one for each shard.
        """
        # Step 1: Query the entire truth table for the part
        all_event_nos = [event_no for subset in event_no_subsets for event_no in subset]
        event_filter = ','.join(map(str, all_event_nos))
        query = f"""
            SELECT t.event_no, t.energy, t.azimuth, t.zenith, t.pid,
                COUNT(DISTINCT s.string || '-' || s.dom_number) AS N_doms
            FROM {truth_table_name} t
            JOIN {source_table} s ON t.event_no = s.event_no
            WHERE t.event_no IN ({event_filter})
            GROUP BY t.event_no
        """
        df_combined = pd.read_sql_query(query, con_source)

        # Step 2: Enhance event numbers
        df_combined['event_no'] = df_combined['event_no'].map(
            lambda x: int(f"{part_no}{x}")
        )
        df_combined['offset'] = np.cumsum(df_combined['N_doms'])

        # Step 3: Split into shards
        shard_tables = []
        for subset in event_no_subsets:
            shard_table = df_combined[df_combined['event_no'].isin(subset)]
            shard_tables.append(pa.Table.from_pandas(shard_table))

        return shard_tables

    def _get_part_no(self, file: str) -> int:
        """
        file (str): get the number of the file (e.g., "merged_part_1.db").
        """
        try:
            file_no = int(file.split('_')[-1].split('.')[0])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid file name format: '{file}' does not contain a numeric identifier.")
        
        return file_no
    
    def _add_enhance_event_no(self, pa_table: pa.Table, part_no: int) -> pa.Table:
        """
        event_no transformed to a unique identifier for the entire dataset.
        """
        if 'event_no' in pa_table.schema.names:
            original_event_no = pa_table['event_no']
            enhanced_event_no = [int(f"{part_no}{event_no.as_py()}") for event_no in original_event_no]

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
                    dest_root: str,
                    source_subdirectory: str,
                    part_no: int,
                    shard_no: int,
                    offset: int,
                    limit: int) -> pd.DataFrame:
        """
        Processes a shard of events from the database, saves PMTfied data to a file, and returns the truth+receipt data.
        """
        # Query the subset of event_no for this shard
        event_no_query = f"""
            SELECT DISTINCT event_no 
            FROM {source_table}
            ORDER BY event_no ASC
            LIMIT {limit} OFFSET {offset}
        """
        event_no_subset = pd.read_sql_query(event_no_query, con_source)['event_no'].tolist()
        
        # NOTE
        # PMTSummariser is the core class to be called for PMTfication
        summariser  = PMTSummariser(
            con_source=con_source,
            source_table=source_table,
            event_no_subset=event_no_subset)
        
        pa_pmtfied = summariser()
        pa_pmtfied = self._add_enhance_event_no(pa_pmtfied, part_no)
        
        dest_dir = os.path.join(dest_root, source_subdirectory, str(part_no))
        os.makedirs(dest_dir, exist_ok=True)

        pmtfied_file = os.path.join(dest_dir, f"PMTfied_{shard_no}.parquet")        
        pq.write_table(pa_pmtfied, pmtfied_file)
        
        veritator = PMTTruthMaker(con_source, source_table, truth_table_name, event_no_subset)
        pa_truth_shard = veritator(part_no, shard_no, int(source_subdirectory))
        pa_truth_shard = self._add_enhance_event_no(pa_truth_shard, part_no)
        
        return pa_truth_shard
        
    def _divide_and_conquer_part(self, 
                            con_source: sql.Connection, 
                            source_table: str, 
                            truth_table_name: str,
                            dest_root: str, 
                            source_subdirectory: str, 
                            part_no: int, 
                            N_events_per_shard: int,
                            N_events_total: int
                            ) -> pd.DataFrame:
        """
        Divides the database events into shards and processes each shard, consolidating the truth+receipt data.
        """
        truth_shards = []
        num_shards = (N_events_total + N_events_per_shard - 1) // N_events_per_shard
        
        for shard_no in range(num_shards):
            offset = shard_no * N_events_per_shard
            limit = min(N_events_per_shard, N_events_total - offset)

            # Process the shard and retrieve the truth+receipt data
            pa_truth_shard = self.pmtfy_shard(
                con_source=con_source,
                source_table=source_table,
                truth_table_name=truth_table_name,
                dest_root=dest_root,
                source_subdirectory=source_subdirectory,
                part_no=part_no,
                shard_no=shard_no + 1,
                offset=offset,
                limit=limit
            )
            
            truth_shards.append(pa_truth_shard)

        consolidated_truth = pa.concat_tables(truth_shards)
        return consolidated_truth
        
    def pmtfy_part(self,
                source_subdirectory: str,
                source_file: str,
                dest_root: str,
                source_table: str,
                N_events_per_shard: int = 2000) -> None:
        part_no = self._get_part_no(source_file)
        con_source = sql.connect(source_file)
        truth_table_name = self._get_truth_table_name_db(con_source)
        N_events_total = self._get_table_event_count(con_source, source_table)
        
        
        consolidated_truth = self._divide_and_conquer_part(
            con_source=con_source,
            source_table=source_table,
            truth_table_name=truth_table_name,
            dest_root=dest_root,
            source_subdirectory=source_subdirectory,
            part_no=part_no,
            N_events_per_shard=N_events_per_shard,
            N_events_total=N_events_total
        )

        dest_subdirectory_path = os.path.join(dest_root, source_subdirectory)
        os.makedirs(dest_subdirectory_path, exist_ok=True)
        consolidated_file = os.path.join(dest_subdirectory_path, f"truth_{part_no}.parquet")
        pq.write_table(consolidated_truth, consolidated_file)

        con_source.close()
    
    ### DEPRECATED ###
    def pmtfy_subdir(self, 
                    source_root: str, 
                    dest_root: str, 
                    subdirectory_name: str, 
                    source_table: str, 
                    N_events_per_shard: int = 2000) -> None:
        """
        Processes each database file in a specific subdirectory and saves results in a mirrored directory structure.
        """
        subdirectory_path = os.path.join(source_root, subdirectory_name)
        if os.path.isdir(subdirectory_path) and subdirectory_name.isdigit():
            # List all files to process in the directory
            files = [f for f in os.listdir(subdirectory_path) if f.endswith('.db')]
            logging.info(f"Found {len(files)} database files in subdirectory {subdirectory_name}.")
            
            for filename in tqdm(files, desc=f"Processing {subdirectory_name}"):
                source_file = os.path.join(subdirectory_path, filename)
                logging.info(f"Starting processing for database file: {filename}")
                # Process each database file within the subdirectory
                self.pmtfy_part(
                    source_subdirectory=subdirectory_name,
                    source_file=source_file,
                    dest_root=dest_root,
                    source_table=source_table,
                    N_events_per_shard=N_events_per_shard,
                )
                logging.info(f"Finished processing for database file: {filename}")
    
    def pmtfy_subdir_parallel(self, 
                            source_root: str, 
                            dest_root: str, 
                            subdirectory_name: str, 
                            source_table: str, 
                            N_events_per_shard: int = 2000, 
                            max_workers: int = 15) -> None:
        """
        Processes each database file in a specific subdirectory using parallel workers 
        and saves results in a mirrored directory structure.
        """
        subdirectory_path = os.path.join(source_root, subdirectory_name)
        if os.path.isdir(subdirectory_path) and subdirectory_name.isdigit():
            files = [f for f in os.listdir(subdirectory_path) if f.endswith('.db') and os.path.isfile(os.path.join(subdirectory_path, f))]
            logging.info(f"Found {len(files)} database files in subdirectory {subdirectory_name}.")
            
            # Adjust max_workers based on the number of files
            actual_workers = min(len(files), max_workers)
            logging.info(f"Using {actual_workers} workers for parallel processing.")
            
            # Process files in parallel
            with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                futures = {}
                for filename in files:
                    source_file = os.path.join(subdirectory_path, filename)
                    futures[executor.submit(self.pmtfy_part, 
                                            source_subdirectory=subdirectory_name,
                                            source_file=source_file,
                                            dest_root=dest_root,
                                            source_table=source_table,
                                            N_events_per_shard=N_events_per_shard)] = filename
                for future in tqdm(as_completed(futures), desc=f"Processing {subdirectory_name}", total=len(files)):
                    filename = futures[future]
                    try:
                        future.result()
                        logging.info(f"Finished processing database file: {filename}")
                    except Exception as e:
                        logging.error(f"Error processing file {filename}: {e}", exc_info=True)
                    
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
    
    max_workers = min(15, cpu_count())
    logging.info(f"Using up to {max_workers} workers.")
    
    # PMTfication logic
    N_events_per_shard = args.N_events_per_shard
    pmtfier = PMTfier(source_table_name, dest_root, N_events_per_shard)
    pmtfier.pmtfy_subdir_parallel(
        source_root=source_root,
        dest_root=dest_root,
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