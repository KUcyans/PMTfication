import pandas as pd
import numpy as np
import pyarrow as pa
import sqlite3 as sql
from typing import List

class PMTTruthMaker:
    def __init__(self, con_source: sql.Connection, source_table: str, truth_table_name: str, event_no_subset: List[int]) -> None:
        self.con_source = con_source
        self.source_table = source_table
        self.truth_table_name = truth_table_name
        self.event_no_subset = event_no_subset

    def __call__(self, part_no: int, shard_index: int, subdirectory_no: int) -> None:
        return self._get_truth_pa_shard(part_no, shard_index, subdirectory_no)
    
    def _get_truth_pa_shard(self, 
                part_no: int, 
                shard_index: int, 
                subdirectory_no: int) -> pa.Table:

        receipt_data = {
            'event_no': self.event_no_subset,
            'subdirectory_no': [subdirectory_no] * len(self.event_no_subset),
            'part_no': [part_no] * len(self.event_no_subset),
            'shard_no': [shard_index] * len(self.event_no_subset)
        }
        df_receipt = pd.DataFrame(receipt_data)

        event_filter = ','.join(map(str, self.event_no_subset))
        truth_query = f"""
            SELECT t.event_no, t.energy, t.azimuth, t.zenith, t.pid,
                COUNT(DISTINCT s.string || '-' || s.dom_number) AS N_doms
            FROM {self.truth_table_name} t
            JOIN {self.source_table} s ON t.event_no = s.event_no
            WHERE t.event_no IN ({event_filter})
            GROUP BY t.event_no
        """
        truth_df = pd.read_sql_query(truth_query, self.con_source)
        truth_df['offset'] = np.cumsum(truth_df['N_doms'])

        truth_df = pd.merge(df_receipt, truth_df, on='event_no', how='inner')
        column_order = [
            'event_no', 'subdirectory_no', 'part_no', 'shard_no', 'N_doms', 'offset',
            'energy', 'azimuth', 'zenith', 'pid'
        ]
        truth_df = truth_df[column_order]
        return pa.Table.from_pandas(truth_df)