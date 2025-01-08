import pyarrow as pa
import pyarrow.compute as pc
import sqlite3 as sql
from typing import List
from pyarrow.compute import SetLookupOptions
import pandas as pd

class PMTTruthMaker:
    _TRUTH_SCHEMA = pa.schema([
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
        ('CascadeFilter_13', pa.int32()),
        ('MuonFilter_13', pa.int32()),
    ])
    
    _HE_DAUGHTER_SCHEMA = pa.schema([
        ('event_no', pa.int32()),
        ('zenith_GNHighestEDaughter', pa.float32()),
        ('azimuth_GNHighestEDaughter', pa.float32()),
        ('energy_GNHighestEDaughter', pa.float32()),
        ('pos_x_GNHighestEDaughter', pa.float32()),
        ('pos_y_GNHighestEDaughter', pa.float32()),
        ('pos_z_GNHighestEDaughter', pa.float32()),
    ])
    """
    IS_DEBUG is a static constant when whose value is True, 
    the merged schema will include the event_no from GNHighestEDaughter.
    """
    IS_DEBUG = False
    
    if IS_DEBUG:
        _MERGED_SCHEMA = pa.schema(
        list(_TRUTH_SCHEMA)
        + [pa.field('HE_daughter_event_no', pa.int32())] 
        + [field for field in _HE_DAUGHTER_SCHEMA if field.name != 'event_no'] 
        )
    else:
        _MERGED_SCHEMA = pa.schema(
        list(_TRUTH_SCHEMA) 
        + [field for field in _HE_DAUGHTER_SCHEMA if field.name != 'event_no']
        )
    
    def __init__(self, 
                 con_source: sql.Connection, 
                 source_table: str, 
                 truth_table_name: str, 
                 HE_dauther_table_name: str) -> None:
        self.con_source = con_source
        self.source_table = source_table
        self.truth_table_name = truth_table_name
        self.HE_dauther_table_name = HE_dauther_table_name

    def __call__(self, subdirectory_no: int, part_no: int, shard_no: int, event_no_subset: List[int]) -> pa.Table:
        receipt_pa = self._build_receipt_pa(subdirectory_no, part_no, shard_no, event_no_subset)
        truth_table = self._get_truth_pa_shard(receipt_pa, event_no_subset)
        HE_daughter_table = self._get_HE_daughter_pa_shard(receipt_pa, event_no_subset)
        return self._merge_tables(truth_table, HE_daughter_table, subdirectory_no, part_no, shard_no)

    def _build_receipt_pa(self, subdirectory_no: int, part_no: int, shard_no: int, event_no_subset: List[int]) -> pa.Table:
        receipt_data = {
            'event_no': event_no_subset,
            'subdirectory_no': [subdirectory_no] * len(event_no_subset),
            'part_no': [part_no] * len(event_no_subset),
            'shard_no': [shard_no] * len(event_no_subset),
        }
        return pa.Table.from_pydict(receipt_data)

    def _get_truth_pa_shard(self, receipt_pa: pa.Table, event_no_subset: List[int]) -> pa.Table:
        truth_query = self._build_truth_query(event_no_subset)
        rows, columns = self._execute_query(truth_query)
        truth_table = self._create_truth_pa_table(rows, columns)
        return self._filter_rows(truth_table, receipt_pa)

    def _get_HE_daughter_pa_shard(self, receipt_pa: pa.Table, event_no_subset: List[int]) -> pa.Table:
        daughter_query = self._build_HE_daughter_query(event_no_subset)
        rows, columns = self._execute_query(daughter_query)
        HE_daughter_table = self._create_HE_daughter_pa_table(rows, columns)
        return self._filter_rows(HE_daughter_table, receipt_pa)

    def _create_truth_pa_table(self, rows: List[tuple], columns: List[str]) -> pa.Table:
        if not rows:
            return pa.Table.from_pydict({field.name: [] for field in PMTTruthMaker._TRUTH_SCHEMA}, schema=PMTTruthMaker._TRUTH_SCHEMA)

        truth_data = {col: [row[i] for row in rows] for i, col in enumerate(columns)}
        truth_data['offset'] = pc.cumulative_sum(pa.array(truth_data['N_doms']))
        return pa.Table.from_pydict(truth_data)
    
    def _create_HE_daughter_pa_table(self, rows: List[tuple], columns: List[str]) -> pa.Table:
        if not rows:
            return pa.Table.from_pydict({field.name: [] for field in PMTTruthMaker._HE_DAUGHTER_SCHEMA}, schema=PMTTruthMaker._HE_DAUGHTER_SCHEMA)

        daughter_data = {col: [row[i] for row in rows] for i, col in enumerate(columns)}
        daughter_table = pa.Table.from_pydict(daughter_data, schema=PMTTruthMaker._HE_DAUGHTER_SCHEMA)

        daughter_table = daughter_table.cast(PMTTruthMaker._HE_DAUGHTER_SCHEMA)

        replacement_values = {
            'zenith_GNHighestEDaughter': -1,
            'azimuth_GNHighestEDaughter': -1,
            'pos_x_GNHighestEDaughter': -1e8,
            'pos_y_GNHighestEDaughter': -1e8,
            'pos_z_GNHighestEDaughter': -1e8
        }

        for column, replacement in replacement_values.items():
            if column in daughter_table.column_names:
                # Handle null
                filled_column = pc.fill_null(daughter_table[column], replacement)
                # Handle NaN
                replaced_column = pc.if_else(
                    pc.is_nan(filled_column),
                    pa.scalar(replacement, type=filled_column.type),
                    filled_column
                )
                daughter_table = daughter_table.set_column(
                    daughter_table.schema.get_field_index(column),
                    column,
                    replaced_column
                )

        return daughter_table

    def _merge_tables(self, truth_table: pa.Table, HE_daughter_table: pa.Table, subdirectory_no: int, part_no: int, shard_no: int) -> pa.Table:
        daughter_event_map = {row['event_no']: row for row in HE_daughter_table.to_pylist()}

        merged_data = {
            'event_no': truth_table['event_no'],
            'subdirectory_no': pa.array([subdirectory_no] * len(truth_table)),
            'part_no': pa.array([part_no] * len(truth_table)),
            'shard_no': pa.array([shard_no] * len(truth_table)),
            **{col: truth_table[col] for col in truth_table.column_names if col not in ['event_no']}
        }
        
        if PMTTruthMaker.IS_DEBUG:
            merged_data['HE_daughter_event_no'] = [
                daughter_event_map.get(event, {}).get('event_no', None) for event in truth_table['event_no'].to_pylist()
            ]
        for col in [
            'zenith_GNHighestEDaughter', 'azimuth_GNHighestEDaughter', 'energy_GNHighestEDaughter',
            'pos_x_GNHighestEDaughter', 'pos_y_GNHighestEDaughter', 'pos_z_GNHighestEDaughter'
        ]:
            merged_data[col] = [
                daughter_event_map.get(event, {}).get(col, None) for event in truth_table['event_no'].to_pylist()
            ]
        return pa.Table.from_pydict(merged_data, schema=PMTTruthMaker._MERGED_SCHEMA)
    
    def _filter_rows(self, table: pa.Table, receipt_pa: pa.Table) -> pa.Table:
        event_no_column_truth_list = table['event_no'].to_pylist()
        event_no_column_receipt_list = receipt_pa['event_no'].to_pylist()

        if not event_no_column_truth_list or not event_no_column_receipt_list:
            return pa.Table.from_pydict({field.name: [] for field in PMTTruthMaker._MERGED_SCHEMA}, schema=PMTTruthMaker._MERGED_SCHEMA)

        lookup_options = SetLookupOptions(value_set=pa.array(event_no_column_receipt_list))
        filtered_rows = pc.is_in(pa.array(event_no_column_truth_list), options=lookup_options)
        return table.filter(filtered_rows)

    def _build_truth_query(self, event_no_subset: List[int]) -> str:
        event_filter = ','.join(map(str, event_no_subset))
        return f"""
            SELECT t.event_no, t.energy, t.azimuth, t.zenith, t.pid, t.CascadeFilter_13, t.MuonFilter_13,
                COUNT(DISTINCT s.string || '-' || s.dom_number) AS N_doms
            FROM {self.truth_table_name} t
            JOIN {self.source_table} s ON t.event_no = s.event_no
            WHERE t.event_no IN ({event_filter})
            GROUP BY t.event_no
        """

    def _build_HE_daughter_query(self, event_no_subset: List[int]) -> str:
        event_filter = ','.join(map(str, event_no_subset))
        return f"""
            SELECT h.event_no, h.zenith_GNHighestEDaughter, h.azimuth_GNHighestEDaughter, h.energy_GNHighestEDaughter,
                h.pos_x_GNHighestEDaughter, h.pos_y_GNHighestEDaughter, h.pos_z_GNHighestEDaughter
            FROM {self.HE_dauther_table_name} h
            WHERE h.event_no IN ({event_filter})
        """

    def _execute_query(self, query: str) -> (List[tuple], List[str]):
        cursor = self.con_source.cursor()
        cursor.execute(query)
        return cursor.fetchall(), [desc[0] for desc in cursor.description]
