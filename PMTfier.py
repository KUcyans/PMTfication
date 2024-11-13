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

from PMT_summariser import PMTSummariser
from PMT_ref_pos_adder import ReferencePositionAdder

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
        
    def _get_truth_pa(self,
                    con_source: sql.Connection,
                    source_table: str,
                    truth_table_name: str,
                    event_no_subset: List[int],
                    subdirectory_no: int,
                    db_file_no: int,
                    shard_index: int) -> pd.DataFrame:

        # Generate receipt data
        receipt_data = {
            'event_no': event_no_subset,
            'subdirectory_no': [subdirectory_no] * len(event_no_subset),
            'db_file_no': [db_file_no] * len(event_no_subset),
            'shard_index': [shard_index] * len(event_no_subset),
            'file_no': [shard_index] * len(event_no_subset)
        }
        df_receipt = pd.DataFrame(receipt_data)

        # Combine queries for unique DOM counts and truth features
        event_filter = ','.join(map(str, event_no_subset))
        query = f"""
            SELECT t.event_no, t.energy, t.azimuth, t.zenith, t.pid,
                COUNT(DISTINCT s.string || '-' || s.dom_number) AS N_doms
            FROM {truth_table_name} t
            JOIN {source_table} s ON t.event_no = s.event_no
            WHERE t.event_no IN ({event_filter})
            GROUP BY t.event_no
        """
        df_combined = pd.read_sql_query(query, con_source)
        df_combined['offset'] = np.cumsum(df_combined['N_doms'])
        df_combined = pd.merge(df_receipt, df_combined, on='event_no', how='inner')

        column_order = [
            'event_no', 'subdirectory_no', 'db_file_no', 'shard_index', 'file_no',
            'N_doms', 'offset',
            'energy', 'azimuth', 'zenith', 'pid'
        ]
        df_combined = df_combined[column_order]
        return df_combined
        
    def _get_subdirectory_no(self, subdirectory: str) -> int:
        """
        subdirectory (str): Name of the subdirectory (e.g., "22010").
        """
        try:
            subdirectory_no = int(subdirectory)
        except ValueError:
            raise ValueError(f"Invalid subdirectory name: '{subdirectory}' is not a numeric value.")
        
        return subdirectory_no

    def _get_db_file_no(self, file: str) -> int:
        """
        file (str): Name of the file (e.g., "merged_part_1.db").
        """
        try:
            # Assuming format like "merged_part_X.db" where X is the file number
            file_no = int(file.split('_')[-1].split('.')[0])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid file name format: '{file}' does not contain a numeric identifier.")
        
        return file_no
        
    def pmtfy_shard(self, 
                    con_source: sql.Connection,
                    source_table: str,
                    truth_table_name: str,
                    dest_root: str,
                    source_subdirectory: str,
                    db_file_no: int,
                    shard_index: int,
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
        
        if not event_no_subset:
            return pd.DataFrame()  # Return an empty DataFrame if no events in this shard
        
        ## NOTE: ReferencePositionAdder adds string and dom_number based on the reference data
        ## The reference is hardcoded in PMT_ref_pos_adder.py
        ref_pos_adder = ReferencePositionAdder(
            con_source=con_source,
            source_table=source_table,
            event_no_subset=event_no_subset,
            tolerance_xy=10,
            tolerance_z=2
        )
        ref_pos_adder()
        
        ##NOTE: PMTSummariser is the core class to be called for PMTfication
        summariser  = PMTSummariser(
            con_source=con_source,
            source_table=source_table,
            event_no_subset=event_no_subset)
        
        pa_pmtfied = summariser()
        
        dest_dir = os.path.join(dest_root, source_subdirectory, str(db_file_no))
        os.makedirs(dest_dir, exist_ok=True)

        pmtfied_file = os.path.join(dest_dir, f"PMTfied_{shard_index}.parquet")        
        pq.write_table(pa_pmtfied, pmtfied_file)
        
        truth_df = self._get_truth_pa(
            con_source,
            source_table=source_table,
            truth_table_name = truth_table_name,
            event_no_subset=event_no_subset,
            subdirectory_no=int(source_subdirectory),
            db_file_no=db_file_no,
            shard_index=shard_index
        )

        return truth_df
        
    def _divide_and_conquer_db(self, 
                            con_source: sql.Connection, 
                            source_table: str, 
                            truth_table_name: str,
                            dest_root: str, 
                            source_subdirectory: str, 
                            db_file_no: int, 
                            N_events_per_shard: int,
                            N_events_total: int
                            ) -> pd.DataFrame:
        """
        Divides the database events into shards and processes each shard, consolidating the truth+receipt data.
        """
        all_shards_df = []
        num_shards = (N_events_total + N_events_per_shard - 1) // N_events_per_shard
        
        for shard_index in range(num_shards):
            offset = shard_index * N_events_per_shard
            limit = min(N_events_per_shard, N_events_total - offset)

            # Process the shard and retrieve the truth+receipt data
            shard_df = self.pmtfy_shard(
                con_source=con_source,
                source_table=source_table,
                truth_table_name=truth_table_name,
                dest_root=dest_root,
                source_subdirectory=source_subdirectory,
                db_file_no=db_file_no,
                shard_index=shard_index + 1,
                offset=offset,
                limit=limit
            )
            
            # Append shard data to the collection
            if not shard_df.empty:
                all_shards_df.append(shard_df)

        consolidated_df = pd.concat(all_shards_df, ignore_index=True)
        return consolidated_df
        
    def pmtfy_db(self,
                source_subdirectory: str,
                source_file: str,
                dest_root: str,
                source_table: str,
                N_events_per_shard: int = 2000) -> None:
        db_file_no = self._get_db_file_no(source_file)
        con_source = sql.connect(source_file)
        truth_table_name = self._get_truth_table_name_db(con_source)
        N_events_total = self._get_table_event_count(con_source, source_table)

        consolidated_df = self._divide_and_conquer_db(
            con_source=con_source,
            source_table=source_table,
            truth_table_name=truth_table_name,
            dest_root=dest_root,
            source_subdirectory=source_subdirectory,
            db_file_no=db_file_no,
            N_events_per_shard=N_events_per_shard,
            N_events_total=N_events_total  # Pass total count down
        )

        dest_subdirectory_path = os.path.join(dest_root, source_subdirectory)
        os.makedirs(dest_subdirectory_path, exist_ok=True)
        consolidated_file = os.path.join(dest_subdirectory_path, f"truth_{db_file_no}.parquet")
        consolidated_pa = pa.Table.from_pandas(consolidated_df)
        pq.write_table(consolidated_pa, consolidated_file)

        con_source.close()
        
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
                self.pmtfy_db(
                    source_subdirectory=subdirectory_name,
                    source_file=source_file,
                    dest_root=dest_root,
                    source_table=source_table,
                    N_events_per_shard=N_events_per_shard,
                )
                logging.info(f"Finished processing for database file: {filename}")

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
    
    # PMTfication logic
    N_events_per_shard = args.N_events_per_shard
    pmtfier = PMTfier(source_table_name, dest_root, N_events_per_shard)
    pmtfier.pmtfy_subdir(
        source_root=source_root,
        dest_root=dest_root,
        subdirectory_name=args.subdirectory_in_number,
        source_table=source_table_name,
        N_events_per_shard=N_events_per_shard
    )
    logging.info("PMTfication completed.")

# Wrap the main function in a try-except block
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)