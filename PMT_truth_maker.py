import pyarrow as pa
import pyarrow.compute as pc
import sqlite3 as sql
from typing import List, Dict
from pyarrow.compute import SetLookupOptions


class PMTTruthMaker:
    _SCHEMA = pa.schema([
            ('event_no', pa.int32()),
            ('subdirectory_no', pa.int32()),
            ('part_no', pa.int32()),
            ('shard_no', pa.int32()),
            ('N_doms', pa.int32()),
            ('offset', pa.int32()),
            ('energy', pa.float32()),
            ('azimuth', pa.float32()),
            ('zenith', pa.float32()),
            ('pid', pa.int32())
        ])

    def __init__(self, con_source: sql.Connection, source_table: str, truth_table_name: str, event_no_subset: List[int]) -> None:
        self.con_source = con_source
        self.source_table = source_table
        self.truth_table_name = truth_table_name
        self.event_no_subset = event_no_subset
        
    def __call__(self, subdirectory_no: int, part_no: int, shard_no: int) -> pa.Table:
        return self._get_truth_pa_shard(subdirectory_no, part_no, shard_no)
        
    def _get_truth_pa_shard(self, subdirectory_no: int, part_no: int, shard_no: int) -> pa.Table:
        # Create receipt data
        receipt_data = {
            'event_no': self.event_no_subset,
            'subdirectory_no': [subdirectory_no] * len(self.event_no_subset),
            'part_no': [part_no] * len(self.event_no_subset),
            'shard_no': [shard_no] * len(self.event_no_subset)
        }
        receipt_pa = pa.Table.from_pydict(receipt_data)

        # SQL Query
        event_filter = ','.join(map(str, self.event_no_subset))
        truth_query = f"""
            SELECT t.event_no, t.energy, t.azimuth, t.zenith, t.pid,
                COUNT(DISTINCT s.string || '-' || s.dom_number) AS N_doms
            FROM {self.truth_table_name} t
            JOIN {self.source_table} s ON t.event_no = s.event_no
            WHERE t.event_no IN ({event_filter})
            GROUP BY t.event_no
        """
        cursor = self.con_source.cursor()
        cursor.execute(truth_query)
        rows = cursor.fetchall()
        if not rows:
            return pa.Table.from_pydict({field.name: [] for field in PMTTruthMaker._SCHEMA}, schema=PMTTruthMaker._SCHEMA)

        columns = [desc[0] for desc in cursor.description]
        truth_data = {col: [row[i] for row in rows] for i, col in enumerate(columns)}
        truth_data['offset'] = pc.cumulative_sum(pa.array(truth_data['N_doms']))
        truth_table = pa.Table.from_pydict(truth_data)

        event_no_column_truth_list = truth_table['event_no'].to_pylist()
        event_no_column_receipt_list = receipt_pa['event_no'].to_pylist()

        if not event_no_column_truth_list or not event_no_column_receipt_list:
            return pa.Table.from_pydict({field.name: [] for field in PMTTruthMaker._SCHEMA}, schema=PMTTruthMaker._SCHEMA)

        lookup_options = SetLookupOptions(value_set=pa.array(event_no_column_receipt_list))
        filtered_rows = pc.is_in(pa.array(event_no_column_truth_list), options=lookup_options)
        matched_truth_table = truth_table.filter(filtered_rows)

        merged_data = {
            'event_no': matched_truth_table['event_no'],
            'subdirectory_no': pa.array([subdirectory_no] * len(matched_truth_table)),
            'part_no': pa.array([part_no] * len(matched_truth_table)),
            'shard_no': pa.array([shard_no] * len(matched_truth_table)),
            **{col: matched_truth_table[col] for col in truth_table.column_names if col not in ['event_no']}
        }

        return pa.Table.from_pydict(merged_data, schema=PMTTruthMaker._SCHEMA)

