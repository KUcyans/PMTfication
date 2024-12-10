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
            ('pid', pa.int32()),
            # ('azimuth_HE_daughter', pa.float32()),
            # ('zenith_HE_daughter', pa.float32()),
            # ('energy_HE_daughter', pa.float32()),
            # ('pos_x_HE_daughter', pa.float32()),
            # ('pos_y_HE_daughter', pa.float32()),
            # ('pos_z_HE_daughter', pa.float32()),
        ])

    def __init__(self, 
                 con_source: sql.Connection, 
                 source_table: str, 
                 truth_table_name: str, 
                 HE_dauther_table_name: str,
                 event_no_subset: List[int]) -> None:
        self.con_source = con_source
        self.source_table = source_table
        self.truth_table_name = truth_table_name
        self.HE_dauther_table_name = HE_dauther_table_name
        self.event_no_subset = event_no_subset
        self.receipt_pa = self._build_receipt_pa()
    
    def _build_receipt_pa(self) -> pa.Table:
        receipt_data = {
            'event_no': self.event_no_subset,
            'subdirectory_no': [0] * len(self.event_no_subset),
            'part_no': [0] * len(self.event_no_subset),
            'shard_no': [0] * len(self.event_no_subset)
        }
        return pa.Table.from_pydict(receipt_data)
        
    def __call__(self, subdirectory_no: int, part_no: int, shard_no: int) -> pa.Table:
        return self._get_truth_pa_shard(subdirectory_no, part_no, shard_no)
    
    def _get_truth_pa_shard(self, subdirectory_no: int, part_no: int, shard_no: int) -> pa.Table:
        truth_query = self._build_truth_query()
        rows, columns = self._execute_query(truth_query)
        truth_table = self._create_truth_pa_table(rows, columns)
        matched_truth_table = self._filter_rows(truth_table)
        return self._merge_tables(matched_truth_table, subdirectory_no, part_no, shard_no)
    
    def _merge_tables(self, matched_truth_table: pa.Table, subdirectory_no: int, part_no: int, shard_no: int) -> pa.Table:
        merged_data = {
            'event_no': matched_truth_table['event_no'],
            'subdirectory_no': pa.array([subdirectory_no] * len(matched_truth_table)),
            'part_no': pa.array([part_no] * len(matched_truth_table)),
            'shard_no': pa.array([shard_no] * len(matched_truth_table)),
            **{col: matched_truth_table[col] for col in matched_truth_table.column_names if col not in ['event_no']}
        }
        return pa.Table.from_pydict(merged_data, schema=PMTTruthMaker._SCHEMA)

    def _filter_rows(self, table: pa.Table) -> pa.Table:
        event_no_column_truth_list = table['event_no'].to_pylist()
        event_no_column_receipt_list = self.receipt_pa['event_no'].to_pylist()

        if not event_no_column_truth_list or not event_no_column_receipt_list:
            return pa.Table.from_pydict({field.name: [] for field in PMTTruthMaker._SCHEMA}, schema=PMTTruthMaker._SCHEMA)

        lookup_options = SetLookupOptions(value_set=pa.array(event_no_column_receipt_list))
        filtered_rows = pc.is_in(pa.array(event_no_column_truth_list), options=lookup_options)
        return table.filter(filtered_rows)
    
    def _create_truth_pa_table(self, rows: List[tuple], columns: List[str]) -> pa.Table:
        if not rows:
            return pa.Table.from_pydict({field.name: [] for field in PMTTruthMaker._SCHEMA}, schema=PMTTruthMaker._SCHEMA)
        
        truth_data = {col: [row[i] for row in rows] for i, col in enumerate(columns)}
        truth_data['offset'] = pc.cumulative_sum(pa.array(truth_data['N_doms']))
        return pa.Table.from_pydict(truth_data)
    
    def _build_truth_query(self) -> str:
        """Construct the SQL query for the truth table."""
        event_filter = ','.join(map(str, self.event_no_subset))
        truth_query = f"""
            SELECT t.event_no, t.energy, t.azimuth, t.zenith, t.pid,
                COUNT(DISTINCT s.string || '-' || s.dom_number) AS N_doms
            FROM {self.truth_table_name} t
            JOIN {self.source_table} s ON t.event_no = s.event_no
            WHERE t.event_no IN ({event_filter})
            GROUP BY t.event_no
        """
        return truth_query
    
    def _execute_query(self, query: str) -> List[Dict]:
        cursor = self.con_source.cursor()
        cursor.execute(query)
        return cursor.fetchall(), [desc[0] for desc in cursor.description]