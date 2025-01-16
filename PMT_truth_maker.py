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
        ('event_time', pa.float32()),
        ('interaction_type', pa.int32()),
        ('elasticity', pa.float32()),
        ('RunID', pa.int64()),
        ('SubrunID', pa.int64()),
        ('EventID', pa.int32()),
        ('SubEventID', pa.int32()),
        ('dbang_decay_length', pa.float32()),
        ('track_length', pa.float32()),
        ('stopped_muon', pa.int32()),
        ('energy_track', pa.float32()),
        ('energy_cascade', pa.float32()),
        ('inelasticity', pa.float32()),
        ('DeepCoreFilter_13', pa.int32()),
        ('CascadeFilter_13', pa.int32()),
        ('MuonFilter_13', pa.int32()),
        ('OnlineL2Filter_17', pa.int32()),
        ('L3_oscNext_bool', pa.int32()),
        ('L4_oscNext_bool', pa.int32()),
        ('L5_oscNext_bool', pa.int32()),
        ('L6_oscNext_bool', pa.int32()),
        ('L7_oscNext_bool', pa.int32()),
        ('Homogenized_QTot', pa.float32()),
        ('MCLabelClassification', pa.int32()),
        ('MCLabelCoincidentMuons', pa.int32()),
        ('MCLabelBgMuonMCPE', pa.int32()),
        ('MCLabelBgMuonMCPECharge', pa.int32()),
    ])
    
    _GNLabel_SCHEMA = pa.schema([
        ('GNLabel_event_no', pa.int32()),
        ('GNLabelTrackEnergyDeposited', pa.float32()),
        ('GNLabelTrackEnergyOnEntrance', pa.float32()),
        ('GNLabelTrackEnergyOnEntrancePrimary', pa.float32()),
        ('GNLabelTrackEnergyDepositedPrimary', pa.float32()),
        ('GNLabelEnergyPrimary', pa.float32()),
        ('GNLabelCascadeEnergyDepositedPrimary', pa.float32()),
        ('GNLabelCascadeEnergyDeposited', pa.float32()),
        ('GNLabelEnergyDepositedTotal', pa.float32()),
        ('GNLabelEnergyDepositedPrimary', pa.float32()),
        ('GNLabelHighestEInIceParticleIsChild', pa.int32()),
        ('GNLabelHighestEInIceParticleDistance', pa.float32()),
        ('GNLabelHighestEInIceParticleEFraction', pa.float32()),
        # ('GNLabelHighestEInIceParticleEOnEntrance', pa.float32()),
        ('GNLabelHighestEDaughterDistance', pa.float32()),
        ('GNLabelHighestEDaughterEFraction', pa.float32()),
        ])
    
    _HighestEInIceParticle_SCHEMA = pa.schema([
        ('HighestEInIceParticle_event_no', pa.int32()),
        ('zenith_GNHighestEInIceParticle', pa.float32()),
        ('azimuth_GNHighestEInIceParticle', pa.float32()),
        ('dir_x_GNHighestEInIceParticle', pa.float32()),
        ('dir_y_GNHighestEInIceParticle', pa.float32()),
        ('dir_z_GNHighestEInIceParticle', pa.float32()),
        ('pos_x_GNHighestEInIceParticle', pa.float32()),
        ('pos_y_GNHighestEInIceParticle', pa.float32()),
        ('pos_z_GNHighestEInIceParticle', pa.float32()),
        ('time_GNHighestEInIceParticle', pa.float32()),
        ('speed_GNHighestEInIceParticle', pa.float32()),
        ('energy_GNHighestEInIceParticle', pa.float32()),
        ])
    
    _HE_DAUGHTER_SCHEMA = pa.schema([
        ('HE_daughter_event_no', pa.int32()),
        ('zenith_GNHighestEDaughter', pa.float32()),
        ('azimuth_GNHighestEDaughter', pa.float32()),
        ('dir_x_GNHighestEDaughter', pa.float32()),
        ('dir_y_GNHighestEDaughter', pa.float32()),
        ('dir_z_GNHighestEDaughter', pa.float32()),
        ('pos_x_GNHighestEDaughter', pa.float32()),
        ('pos_y_GNHighestEDaughter', pa.float32()),
        ('pos_z_GNHighestEDaughter', pa.float32()),
        ('time_GNHighestEDaughter', pa.float32()),
        ('speed_GNHighestEDaughter', pa.float32()),
        ('energy_GNHighestEDaughter', pa.float32()),
    ])
    """
    IS_DEBUG is a static constant when whose value is True, 
    the merged schema will include the event_no from GNHighestEDaughter.
    """
    IS_DEBUG = False
    
    if IS_DEBUG:
        _MERGED_SCHEMA = pa.schema(
        list(_TRUTH_SCHEMA)
        + [field for field in _GNLabel_SCHEMA]
        + [field for field in _HighestEInIceParticle_SCHEMA]
        + [field for field in _HE_DAUGHTER_SCHEMA]
        )
    else:
        _MERGED_SCHEMA = pa.schema(
        list(_TRUTH_SCHEMA)
        + [field for field in _GNLabel_SCHEMA if field.name != 'GNLabel_event_no']
        + [field for field in _HighestEInIceParticle_SCHEMA if field.name != 'HighestEInIceParticle_event_no']
        + [field for field in _HE_DAUGHTER_SCHEMA if field.name != 'HE_daughter_event_no']
        )
    
    def __init__(self, 
                 con_source: sql.Connection, 
                 source_table: str, 
                 truth_table_name: str,
                 HighestEInIceParticle_table_name: str, 
                 HE_dauther_table_name: str) -> None:
        self.con_source = con_source
        self.source_table = source_table
        self.truth_table_name = truth_table_name
        self.HighestEInIceParticle_table_name = HighestEInIceParticle_table_name
        self.HE_dauther_table_name = HE_dauther_table_name
        self._build_nan_replacement()

    def __call__(self, subdirectory_no: int, part_no: int, shard_no: int, event_no_subset: List[int]) -> pa.Table:
        receipt_pa = self._build_receipt_pa(subdirectory_no, part_no, shard_no, event_no_subset)
        truth_table = self._get_truth_pa_shard(receipt_pa, event_no_subset)
        
        GNLabel_table = self._get_GNLabel_pa_shard(receipt_pa, event_no_subset)
        HighestEInIceParticle_table = self._get_HighestEInIceParticle_pa_shard(receipt_pa, event_no_subset)
        HE_daughter_table = self._get_HE_daughter_pa_shard(receipt_pa, event_no_subset)
        
        return self._merge_tables(truth_table, 
                                  GNLabel_table,
                                  HighestEInIceParticle_table,
                                  HE_daughter_table, 
                                  subdirectory_no, part_no, shard_no)

    def _build_receipt_pa(self, subdirectory_no: int, part_no: int, shard_no: int, event_no_subset: List[int]) -> pa.Table:
        receipt_data = {
            'event_no': event_no_subset,
            'subdirectory_no': [subdirectory_no] * len(event_no_subset),
            'part_no': [part_no] * len(event_no_subset),
            'shard_no': [shard_no] * len(event_no_subset),
        }
        return pa.Table.from_pydict(receipt_data)
    
    def _merge_tables(self, 
                  truth_table: pa.Table, 
                  GNLabel_table: pa.Table,
                  HighestEInIceParticle_table: pa.Table,
                  HE_daughter_table: pa.Table, 
                  subdirectory_no: int, part_no: int, shard_no: int) -> pa.Table:
        merged_data = {
            'event_no': truth_table['event_no'],
            'subdirectory_no': pa.array([subdirectory_no] * len(truth_table)),
            'part_no': pa.array([part_no] * len(truth_table)),
            'shard_no': pa.array([shard_no] * len(truth_table)),
            **{col: truth_table[col] for col in truth_table.column_names if col != 'event_no'},
            **{col_GN: GNLabel_table[col_GN] for col_GN in GNLabel_table.column_names if col_GN != 'event_no'},
            **{col_HE: HighestEInIceParticle_table[col_HE] for col_HE in HighestEInIceParticle_table.column_names if col_HE != 'event_no'},
            **{col_HE_daughter: HE_daughter_table[col_HE_daughter] for col_HE_daughter in HE_daughter_table.column_names if col_HE_daughter != ' event_no'},
        }
        return pa.Table.from_pydict(merged_data, schema=PMTTruthMaker._MERGED_SCHEMA)

    def _filter_rows(self, table: pa.Table, receipt_pa: pa.Table, event_no_column: str) -> pa.Table:
        event_no_column_truth_list = table[event_no_column].to_pylist()
        event_no_column_receipt_list = receipt_pa['event_no'].to_pylist()

        if not event_no_column_truth_list or not event_no_column_receipt_list:
            return pa.Table.from_pydict({field.name: [] for field in PMTTruthMaker._MERGED_SCHEMA}, schema=PMTTruthMaker._MERGED_SCHEMA)

        lookup_options = SetLookupOptions(value_set=pa.array(event_no_column_receipt_list))
        filtered_rows = pc.is_in(pa.array(event_no_column_truth_list), options=lookup_options)
        return table.filter(filtered_rows)
    
    # --------- TABLE SHARD GETTERS ---------
    def _get_truth_pa_shard(self, receipt_pa: pa.Table, event_no_subset: List[int]) -> pa.Table:
        truth_query = self._build_truth_query(event_no_subset)
        rows, columns = self._execute_query(truth_query)
        truth_table = self._create_truth_pa_table(rows, columns)
        return self._filter_rows(truth_table, receipt_pa, 'event_no')

    def _get_GNLabel_pa_shard(self, receipt_pa: pa.Table, event_no_subset: List[int]) -> pa.Table:
        query = self._build_GNLabel_query(event_no_subset)
        rows, columns = self._execute_query(query)
        # table = self._create_GNLabel_pa_table(rows, columns)
        table = self._create_trailing_pa_table(rows, columns, PMTTruthMaker._GNLabel_SCHEMA, self.nan_replacement_GNLabel)
        return self._filter_rows(table, receipt_pa, 'GNLabel_event_no')
    
    def _get_HighestEInIceParticle_pa_shard(self, receipt_pa: pa.Table, event_no_subset: List[int]) -> pa.Table:
        query = self._build_HighestEInIceParticle_query(event_no_subset)
        rows, columns = self._execute_query(query)
        # table = self._create_HighestEInIceParticle_pa_table(rows, columns)
        table = self._create_trailing_pa_table(rows, columns, PMTTruthMaker._HighestEInIceParticle_SCHEMA, self.nan_replacement_HighestEInIceParticle)
        return self._filter_rows(table, receipt_pa, 'HighestEInIceParticle_event_no')
    
    def _get_HE_daughter_pa_shard(self, receipt_pa: pa.Table, event_no_subset: List[int]) -> pa.Table:
        query = self._build_HE_daughter_query(event_no_subset)
        rows, columns = self._execute_query(query)
        # table = self._create_HE_daughter_pa_table(rows, columns)
        table = self._create_trailing_pa_table(rows, columns, PMTTruthMaker._HE_DAUGHTER_SCHEMA, self.nan_replacement_HE_daughter)
        return self._filter_rows(table, receipt_pa, 'HE_daughter_event_no')
    
    # --------- TABLE BUILDERS ---------
    def _create_truth_pa_table(self, rows: List[tuple], columns: List[str]) -> pa.Table:
        if not rows:
            return pa.Table.from_pydict({field.name: [] for field in PMTTruthMaker._TRUTH_SCHEMA}, schema=PMTTruthMaker._TRUTH_SCHEMA)

        truth_data = {col: [row[i] for row in rows] for i, col in enumerate(columns)}
        truth_data['offset'] = pc.cumulative_sum(pa.array(truth_data['N_doms']))
        return pa.Table.from_pydict(truth_data)
    
    def _create_trailing_pa_table(self, rows: List[tuple], columns: List[str], schema: pa.Schema, nan_replacement: dict) -> pa.Table:
        if not rows:
            return pa.Table.from_pydict({field.name: [] for field in schema}, schema=schema)
        
        data = {col: [row[i] for row in rows] for i, col in enumerate(columns)}
        table = pa.Table.from_pydict(data, schema=schema)
        
        for column, replacement in nan_replacement.items():
            if column in table.column_names:
                filled_column = pc.fill_null(table[column], replacement)
                replaced_column = pc.if_else(
                    pc.is_nan(filled_column),
                    pa.scalar(replacement, type=filled_column.type),
                    filled_column
                )
                table = table.set_column(table.schema.get_field_index(column), column, replaced_column)
                
        return table
    
    def _build_nan_replacement(self) -> None:
        self.nan_replacement_GNLabel = {
            'GNLabelTrackEnergyDeposited': -1,
            'GNLabelTrackEnergyOnEntrance': -1,
            'GNLabelTrackEnergyOnEntrancePrimary': -1,
            'GNLabelTrackEnergyDepositedPrimary': -1,
            'GNLabelEnergyPrimary': -1,
            'GNLabelCascadeEnergyDepositedPrimary': -1,
            'GNLabelCascadeEnergyDeposited': -1,
            'GNLabelEnergyDepositedTotal': -1,
            'GNLabelEnergyDepositedPrimary': -1,
            'GNLabelHighestEInIceParticleDistance': -1e8,
            'GNLabelHighestEDaughterDistance': -1e8,
        }
        
        self.nan_replacement_HighestEInIceParticle = {
            'zenith_GNHighestEInIceParticle': -1,
            'azimuth_GNHighestEInIceParticle': -1,
            'dir_x_GNHighestEInIceParticle': 0,
            'dir_y_GNHighestEInIceParticle': 0,
            'dir_z_GNHighestEInIceParticle': 0,
            'pos_x_GNHighestEInIceParticle': -1e8,
            'pos_y_GNHighestEInIceParticle': -1e8,
            'pos_z_GNHighestEInIceParticle': -1e8,
            'time_GNHighestEInIceParticle': -1,
            'speed_GNHighestEInIceParticle': -1,
            'energy_GNHighestEInIceParticle': -1,
            }
        
        self.nan_replacement_HE_daughter = {
            'zenith_GNHighestEDaughter': -1,
            'azimuth_GNHighestEDaughter': -1,
            'dir_x_GNHighestEDaughter': 0,
            'dir_y_GNHighestEDaughter': 0,
            'dir_z_GNHighestEDaughter': 0,
            'pos_x_GNHighestEDaughter': -1e8,
            'pos_y_GNHighestEDaughter': -1e8,
            'pos_z_GNHighestEDaughter': -1e8,
            'time_GNHighestEDaughter': -1,
            'speed_GNHighestEDaughter': -1,
            'energy_GNHighestEDaughter': -1,
        }
    
    # --------- QUERY BUILDERS ---------
    def _build_truth_query(self, event_no_subset: List[int]) -> str:
        event_filter = ','.join(map(str, event_no_subset))
        return f"""
            SELECT 
            t.event_no, 
            t.energy, 
            t.azimuth, 
            t.zenith, 
            t.pid, 
            t.event_time,
            t.interaction_type,
            t.elasticity,
            t.RunID,
            t.SubrunID,
            t.EventID,
            t.SubEventID,
            t.dbang_decay_length,
            t.track_length,
            t.stopped_muon,
            t.energy_track,
            t.energy_cascade,
            t.inelasticity,
            t.DeepCoreFilter_13,
            t.CascadeFilter_13,
            t.MuonFilter_13,
            t.OnlineL2Filter_17,
            t.L3_oscNext_bool,
            t.L4_oscNext_bool,
            t.L5_oscNext_bool,
            t.L6_oscNext_bool,
            t.L7_oscNext_bool,
            t.Homogenized_QTot,
            t.MCLabelClassification,
            t.MCLabelCoincidentMuons,
            t.MCLabelBgMuonMCPE,
            t.MCLabelBgMuonMCPECharge,
                COUNT(DISTINCT s.string || '-' || s.dom_number) AS N_doms
            FROM {self.truth_table_name} t
            JOIN {self.source_table} s ON t.event_no = s.event_no
            WHERE t.event_no IN ({event_filter})
            GROUP BY t.event_no
        """

    def _build_GNLabel_query(self, event_no_subset: List[int]) -> str:
        event_filter = ','.join(map(str, event_no_subset))
        #g.GNLabelHighestEInIceParticleEOnEntrance, # missing in debuggin data sample
            # g.GNLabelHighestEInIceParticleEOnEntrance,
        return f"""
            SELECT
            g.event_no AS GNLabel_event_no, -- Alias event_no as GNLabel_event_no
            g.GNLabelTrackEnergyDeposited,
            g.GNLabelTrackEnergyOnEntrance,
            g.GNLabelTrackEnergyOnEntrancePrimary,
            g.GNLabelTrackEnergyDepositedPrimary,
            g.GNLabelEnergyPrimary,
            g.GNLabelCascadeEnergyDepositedPrimary,
            g.GNLabelCascadeEnergyDeposited,
            g.GNLabelEnergyDepositedTotal,
            g.GNLabelEnergyDepositedPrimary,
            g.GNLabelHighestEInIceParticleIsChild,
            g.GNLabelHighestEInIceParticleDistance,
            g.GNLabelHighestEInIceParticleEFraction,
            
            g.GNLabelHighestEDaughterDistance,
            g.GNLabelHighestEDaughterEFraction
            FROM {self.truth_table_name} g
            WHERE g.event_no IN ({event_filter})
        """
    def _build_HighestEInIceParticle_query(self, event_no_subset: List[int]) -> str:
        event_filter = ','.join(map(str, event_no_subset))
        return f"""
            SELECT 
            h.event_no AS HighestEInIceParticle_event_no, -- Alias event_no as HighestEInIceParticle_event_no
            h.zenith_GNHighestEInIceParticle, 
            h.azimuth_GNHighestEInIceParticle, 
            h.energy_GNHighestEInIceParticle,
            h.pos_x_GNHighestEInIceParticle, 
            h.pos_y_GNHighestEInIceParticle, 
            h.pos_z_GNHighestEInIceParticle,
            h.dir_x_GNHighestEInIceParticle,
            h.dir_y_GNHighestEInIceParticle,
            h.dir_z_GNHighestEInIceParticle,
            h.time_GNHighestEInIceParticle,
            h.speed_GNHighestEInIceParticle
            FROM {self.HighestEInIceParticle_table_name} h
            WHERE h.event_no IN ({event_filter})
        """
    
    def _build_HE_daughter_query(self, event_no_subset: List[int]) -> str:
        event_filter = ','.join(map(str, event_no_subset))
        return f"""
            SELECT 
            h.event_no AS HE_daughter_event_no, -- Alias event_no as HE_daughter_event_no
            h.zenith_GNHighestEDaughter, 
            h.azimuth_GNHighestEDaughter, 
            h.energy_GNHighestEDaughter,
            h.pos_x_GNHighestEDaughter, 
            h.pos_y_GNHighestEDaughter, 
            h.pos_z_GNHighestEDaughter,
            h.dir_x_GNHighestEDaughter,
            h.dir_y_GNHighestEDaughter,
            h.dir_z_GNHighestEDaughter,
            h.time_GNHighestEDaughter,
            h.speed_GNHighestEDaughter
            FROM {self.HE_dauther_table_name} h
            WHERE h.event_no IN ({event_filter})
        """

    def _execute_query(self, query: str) -> (List[tuple], List[str]):
        cursor = self.con_source.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return rows, columns
