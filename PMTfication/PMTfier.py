"""
Main manager script for the PMTfication pipeline.
Reads event data, summarises feature data, generate corresponding truth data.
Delegates processing to PMTTruthMaker and PMTSummariser 
"""

import sqlite3 as sql
import os
from os import getenv
import time

from typing import List

import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm
import logging

from concurrent.futures import as_completed, ProcessPoolExecutor
from multiprocessing import cpu_count

from PMT_summariser import PMTSummariser
from PMT_truth_maker import PMTTruthMaker
from PMT_truth_from_summary import PMTTruthFromSummary
from SummaryMode import SummaryMode
from tabulate import tabulate

class PMTfier:
    def __init__(self, 
                source_root: str,
                source_subdirectory: str,
                source_table: str,
                dest_root: str, 
                N_events_per_shard: int,
                summary_mode: SummaryMode = SummaryMode.CLASSIC) -> None:
        self.source_root = source_root
        self.source_table = source_table
        self.source_subdirectory = source_subdirectory
        self.dest_root = dest_root
        self.N_events_per_shard = N_events_per_shard
        self.summary_mode = summary_mode
        
        self.signal_or_noise_name = os.path.basename(os.path.normpath(self.dest_root))
        self.subdir_tag = self._get_subdir_tag()
        self.truth_table_name = self._get_truth_table_name_db()
        self.HighestEInIceParticle_table_name = "GNHighestEInIceParticle" # table name for the highest energy in-ice particle
        self.HE_dauther_table_name = "GNHighestEDaughter" # table name for the highest energy daughter particle
        self.MC_weight_dict_table_name = "MCWeightDict" # table name for the MC weight dictionary
        
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
            cursor = conn.cursor()
            cursor.execute(query)
            batch = [row[0] for row in cursor.fetchall()]
            if not batch:
                break
            last_event_no = batch[-1]
            yield batch
    
    def _get_truth_table_name_db(self) -> str:
        source_dir = os.path.join(self.source_root, self.source_subdirectory)
        
        db_files = [
            f for f in os.listdir(source_dir)
            if f.startswith('merged_part_') and f.endswith('.db')
        ]
        if not db_files:
            raise FileNotFoundError(f"No database files found in: {source_dir}")

        def extract_part_number(fname: str) -> int:
            try:
                return int(fname.split('_')[-1].split('.')[0])
            except Exception:
                return float('inf')  # Push unparseable names to the end

        # Sort db files by part number
        db_files.sort(key=extract_part_number)

        first_db_file = os.path.join(source_dir, db_files[0])
        con_source = sql.connect(first_db_file)
        cur_source = con_source.cursor()
        cur_source.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur_source.fetchall()]
        con_source.close()

        if 'truth' in tables:
            return 'truth'
        elif 'Truth' in tables:
            return 'Truth'
        else:
            raise ValueError(f"Neither 'truth' nor 'Truth' table exists in: {first_db_file}")

    def _get_subdir_tag(self) -> int:
        if self.signal_or_noise_name not in ['Snowstorm', 'Corsika']:
            raise ValueError(f"Invalid destination root: {self.dest_root}.")
        if self.signal_or_noise_name == 'Snowstorm':
            # "22012" -> 12
            # "22017" -> 17
            subdir_tag = int(self.source_subdirectory[-2:])
        elif self.signal_or_noise_name == 'Corsika':
            # "0002000-0002999" -> 20
            # "0003000-0003999" -> 30
            range_start = self.source_subdirectory.split('-')[0]  # "0002000"
            range_end = self.source_subdirectory.split('-')[1]  # "0002999"
            
            first_half_subdir_no = range_start[3:5]  # "20"
            second_half_subdir_no = range_end[3:5]  # "29"
            
            combined_subdir_no = int(first_half_subdir_no + second_half_subdir_no)
            subdir_tag = combined_subdir_no
        else:
            raise ValueError(f"Invalid destination root: {self.dest_root}.")
        return subdir_tag
            
    def _get_part_no(self, file: str) -> int:
        """
        file (str): get the number of the file (e.g., 1 for "merged_part_1.db").
        """
        try:
            file_no = int(file.split('_')[-1].split('.')[0])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid file name format: '{file}' does not contain a numeric identifier.")
        
        return file_no
    
    def _add_enhance_event_no(self, pa_table: pa.Table, part_no: int) -> pa.Table:
        """
        (1) (2)  (3)  (4)
        1   12  0027 00012345 
        
        (1)snowstorm(1) or corsika(2)
        (2)subdir_tag
        (3)part_no
        (4)original event_no
        """
        if self.signal_or_noise_name == 'Snowstorm':
            signal_or_noise_tag = "1"
        elif self.signal_or_noise_name == 'Corsika':
            signal_or_noise_tag = "2"
        else:
            signal_or_noise_tag = "0"
            
        if 'event_no' in pa_table.schema.names:
            original_event_no = pa_table['event_no']#.cast(pa.int32())
            enhanced_event_no = [
                int(f"{signal_or_noise_tag}{self.subdir_tag:02}{part_no:04}{event_no.as_py():08}")
                for event_no in original_event_no
            ]

            pa_table = pa_table.remove_column(pa_table.schema.get_field_index('event_no'))
            pa_table = pa_table.append_column(
                'original_event_no', pa.array(original_event_no, type=pa.int32())
                )
            pa_table = pa_table.append_column(
                'event_no', pa.array(enhanced_event_no, type=pa.int64())
                )

            reordered_columns = ['event_no', 'original_event_no'] + [
                name for name in pa_table.schema.names if name not in ('event_no', 'original_event_no')
            ]
            pa_table = pa_table.select(reordered_columns)

        return pa_table
        
    def pmtfy_shard(self, 
                    con_source: sql.Connection,
                    part_no: int,
                    shard_no: int,
                    truth_maker: PMTTruthMaker,
                    event_batch: List[int]) -> pa.Table:
        dest_dir = os.path.join(self.dest_root, self.source_subdirectory, str(part_no))
        os.makedirs(dest_dir, exist_ok=True)
        
        # NOTE
        # PMTSummariser is the core class to be called for PMTfication
        pa_pmtfied = PMTSummariser(
            con_source=con_source,
            source_table=self.source_table,
            event_no_subset=event_batch,
            summary_mode=self.summary_mode
        )()
        pa_pmtfied = self._add_enhance_event_no(pa_pmtfied, part_no)
        dest_dir = os.path.join(self.dest_root, self.source_subdirectory, str(part_no))
        pmtfied_file = os.path.join(dest_dir, f"PMTfied_{shard_no}.parquet")
        pq.write_table(pa_pmtfied, pmtfied_file)
        
        # size_MB = self.get_file_size_MB(pmtfied_file)
        # logging.info(f"PMTfied_{shard_no}.parquet size: {size_MB:.2f} MB")
        
        summary_derived_truth = PMTTruthFromSummary(pa_pmtfied)()
        
        # NOTE
        # PMT truth table for this shard is created by PMTTruthMaker and returned
        pa_truth_shard = truth_maker(subdirectory_no=int(self.subdir_tag),
                                     part_no=part_no,
                                     shard_no=shard_no, 
                                     event_no_subset=event_batch,
                                     summary_derived_truth_table = summary_derived_truth)
        pa_truth_shard = self._add_enhance_event_no(pa_truth_shard, part_no)
        
        return pa_truth_shard
        
    def _divide_and_conquer_part(self, 
                            con_source: sql.Connection,
                            part_no: int,
                            truth_maker: PMTTruthMaker) -> pa.Table:
        truth_shards = []
        
        event_no_batches = self._get_event_no_batches(con_source, self.source_table, self.N_events_per_shard)
        for shard_no, event_batch in enumerate(event_no_batches, start=1):
            # logging.info(f"Processing shard {shard_no} of part {part_no} in subdirectory {self.source_subdirectory}.")
            start_time = time.time()
            # logging.info(f"Initiating processing shard {shard_no} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            pa_truth_shard = self.pmtfy_shard(
                con_source=con_source,
                part_no=part_no,
                shard_no=shard_no,
                truth_maker=truth_maker,
                event_batch=event_batch)
            truth_shards.append(pa_truth_shard)
            # logging.info(f"Finishing processing shard {shard_no} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            end_time = time.time()
            elapsed_time = end_time - start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            logging.info(f"Elapsed time for shard {shard_no}: {int(hours)}h {int(minutes)}m {int(seconds)}s")

        return pa.concat_tables(truth_shards)
        
    

    def pmtfy_part(self, source_part_file: str) -> None:
        part_no = self._get_part_no(source_part_file)
        source_size_MB = self.get_file_size_MB(source_part_file)
        logging.info(f"Source part file size: {source_size_MB:.2f} MB")
        
        con_source = sql.connect(source_part_file)
        
        truth_maker = PMTTruthMaker(
            con_source=con_source, 
            source_table=self.source_table, 
            truth_table_name=self.truth_table_name, 
            HighestEInIceParticle_table_name=self.HighestEInIceParticle_table_name,
            HE_dauther_table_name=self.HE_dauther_table_name,
            MC_weight_dict_table_name=self.MC_weight_dict_table_name
        )

        consolidated_truth = self._divide_and_conquer_part(
            con_source=con_source,
            part_no=part_no,
            truth_maker=truth_maker
        )

        dest_subdirectory_path = os.path.join(self.dest_root, self.source_subdirectory)
        os.makedirs(dest_subdirectory_path, exist_ok=True)
        consolidated_file = os.path.join(dest_subdirectory_path, f"truth_{part_no}.parquet")
        pq.write_table(consolidated_truth, consolidated_file)
        
        truth_file_size_MB = self.get_file_size_MB(consolidated_file)

        dest_dir = os.path.join(self.dest_root, self.source_subdirectory, str(part_no))
        shard_files = [
            f for f in os.listdir(dest_dir)
            if os.path.isfile(os.path.join(dest_dir, f)) and f.endswith('.parquet') and 'PMTfied' in f
        ]
        
        shard_sizes = []
        for f in shard_files:
            full_path = os.path.join(dest_dir, f)
            size_MB = self.get_file_size_MB(full_path)
            shard_sizes.append((f, f"{size_MB:.2f} MB"))

        total_shards_MB = sum(float(size.replace(" MB", "")) for _, size in shard_sizes)
        avg_size_MB = total_shards_MB / len(shard_sizes) if shard_sizes else 0

        # Construct and log the table
        table_data = shard_sizes + [
            ("Total PMTfied Size", f"{total_shards_MB:.2f} MB"),
            ("Avg PMTfied Shard Size", f"{avg_size_MB:.2f} MB"),
            ("Truth File Size", f"{truth_file_size_MB:.2f} MB"),
        ]
        table_str = tabulate(table_data, headers=["File", "Size"], tablefmt="pretty")
        logging.info(f"\n{table_str}")

        con_source.close()

    
    def pmtfy_subdir_parallel(self, max_workers: int = 5) -> None:
        """
        Processes each database file in a specific subdirectory using parallel workers 
        and saves results in a mirrored directory structure.
        """
        subdirectory_path = os.path.join(self.source_root, self.source_subdirectory)
        file_parts = [f for f in os.listdir(subdirectory_path) if f.endswith('.db') and os.path.isfile(os.path.join(subdirectory_path, f))]
        logging.info(f"Found {len(file_parts)} database files in subdirectory {self.source_subdirectory}.")
        if max_workers is None:
            max_workers = min(len(file_parts), cpu_count() // 2)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.pmtfy_part, 
                                source_part_file=os.path.join(subdirectory_path, file_part)): file_part
                for file_part in file_parts
            }
            for future in tqdm(as_completed(futures), desc=f"Processing {self.source_subdirectory}", total=len(file_parts)):
                filename = futures[future]
                try:
                    future.result()
                    logging.info(f"Finished processing database file: {filename}")
                except Exception as e:
                    logging.error(f"Error processing file {filename}: {str(e)}", exc_info=True)
                    
    def get_file_size_MB(self, path):
        return os.path.getsize(path) / (1024 * 1024)
    
    def get_total_dir_size_MB(self, dir_path: str) -> float:
        total = 0
        for file in os.listdir(dir_path):
            full_path = os.path.join(dir_path, file)
            if os.path.isfile(full_path) and file.endswith(".parquet"):
                total += os.path.getsize(full_path)
        return total / (1024 * 1024)
