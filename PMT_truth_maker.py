import pyarrow as pa
import pyarrow.compute as pc
import sqlite3 as sql
from typing import List, Dict

class PMTTruthMaker:
    _SCHEMA = None

    def __init__(self, con_source: sql.Connection, source_table: str, truth_table_name: str, event_no_subset: List[int]) -> None:
        self.con_source = con_source
        self.source_table = source_table
        self.truth_table_name = truth_table_name
        self.event_no_subset = event_no_subset

        if PMTTruthMaker._SCHEMA is None:
            PMTTruthMaker._SCHEMA = self._build_schema()

    def __call__(self, part_no: int, shard_no: int, subdirectory_no: int) -> pa.Table:
        return self._get_truth_pa_shard(part_no, shard_no, subdirectory_no)
    
    def _get_truth_pa_shard(self, 
                            part_no: int, 
                            shard_no: int, 
                            subdirectory_no: int) -> pa.Table:
        receipt_data = {
            'event_no': self.event_no_subset,
            'subdirectory_no': [subdirectory_no] * len(self.event_no_subset),
            'part_no': [part_no] * len(self.event_no_subset),
            'shard_no': [shard_no] * len(self.event_no_subset)
        }
        receipt_pa = pa.Table.from_pydict(receipt_data)

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
        columns = [desc[0] for desc in cursor.description]
        truth_table = pa.Table.from_arrays(
            [pa.array([row[i] for row in rows]) for i in range(len(columns))],
            names=columns
        )

        offset = pc.cumulative_sum(truth_table['N_doms'])
        truth_table = truth_table.append_column('offset', offset)

        merged_data = {field.name: [] for field in PMTTruthMaker._SCHEMA}
        for row in receipt_pa.to_pydict()['event_no']:
            if row in truth_table['event_no']:
                merged_row = {
                    'event_no': row,
                    'subdirectory_no': subdirectory_no,
                    'part_no': part_no,
                    'shard_no': shard_no,
                    **{col: truth_table[col][row] for col in truth_table.column_names},
                }
                for key, value in merged_row.items():
                    merged_data[key].append(value)

        return pa.Table.from_pydict(merged_data, schema=PMTTruthMaker._SCHEMA)

    def _build_schema(self) -> pa.Schema:
        return pa.schema([
            ('event_no', pa.int64()),
            ('subdirectory_no', pa.int32()),
            ('part_no', pa.int32()),
            ('shard_no', pa.int32()),
            ('N_doms', pa.int32()),
            ('offset', pa.int64()),
            ('energy', pa.float64()),
            ('azimuth', pa.float64()),
            ('zenith', pa.float64()),
            ('pid', pa.int32())
        ])
