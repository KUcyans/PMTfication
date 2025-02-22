import pyarrow as pa
import pyarrow.compute as pc
import sqlite3 as sql
from typing import List
from pyarrow.compute import SetLookupOptions
from PMT_truth_entity import TruthEntity

class PMTTruthMaker:
    def __init__(self, 
                 con_source: sql.Connection, 
                 source_table: str, 
                 truth_table_name: str,
                 HighestEInIceParticle_table_name: str, 
                 HE_dauther_table_name: str,
                MC_weight_dict_table_name: str
                 ) -> None:
        self.con_source = con_source
        self.source_table = source_table
        self.truth_table_name = truth_table_name
        self.HighestEInIceParticle_table_name = HighestEInIceParticle_table_name
        self.HE_dauther_table_name = HE_dauther_table_name
        self.MC_weight_dict_table_name = MC_weight_dict_table_name
        
        self._build_schema()
        self._build_nan_replacement()

    def __call__(self, subdirectory_no: int, part_no: int, shard_no: int, event_no_subset: List[int]) -> pa.Table:
        receipt_pa = self._build_receipt_pa(subdirectory_no, part_no, shard_no, event_no_subset)
        
        
        truth_table = self._get_pa_shard(
            receipt_pa=receipt_pa,
            event_no_subset=event_no_subset,
            schema_name="TRUTH",
            event_no_column="event_no",
            build_query_func=self._build_truth_query
        )
        
        GNLabel_table = self._get_pa_shard(
            receipt_pa=receipt_pa,
            event_no_subset=event_no_subset,
            schema_name="GNLabel",
            event_no_column="GNLabel_event_no",
            build_query_func=self._build_GNLabel_query
        )
        
        HighestEInIceParticle_table = self._get_pa_shard(
            receipt_pa=receipt_pa,
            event_no_subset=event_no_subset,
            schema_name="HighestEInIceParticle",
            event_no_column="HighestEInIceParticle_event_no",
            build_query_func=self._build_HighestEInIceParticle_query
        )
        
        HE_daughter_table = self._get_pa_shard(
            receipt_pa=receipt_pa,
            event_no_subset=event_no_subset,
            schema_name="HE_DAUGHTER",
            event_no_column="HE_daughter_event_no",
            build_query_func=self._build_HE_daughter_query
        )
        
        MCWeightDict_table = self._get_pa_shard(
            receipt_pa=receipt_pa,
            event_no_subset=event_no_subset,
            schema_name="MCWeightDict",
            event_no_column="MCWeightDict_event_no",
            build_query_func=self._build_MCWeightDict_query
        )
        
        return self._merge_tables(truth_table = truth_table, 
                                GNLabel_table = GNLabel_table, 
                                HighestEInIceParticle_table = HighestEInIceParticle_table, 
                                HE_daughter_table = HE_daughter_table,
                                MCWeightDict_table = MCWeightDict_table,
                                subdirectory_no = subdirectory_no, 
                                part_no = part_no, 
                                shard_no = shard_no)


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
                  MCWeightDict_table: pa.Table,
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
            **{col_MCWeightDict: MCWeightDict_table[col_MCWeightDict] for col_MCWeightDict in MCWeightDict_table.column_names if col_MCWeightDict != 'event_no'},
        }
        return pa.Table.from_pydict(merged_data, schema=self._MERGED_SCHEMA)

    def _filter_rows(self, table: pa.Table, receipt_pa: pa.Table, event_no_column: str) -> pa.Table:
        event_no_column_truth_list = table[event_no_column].to_pylist()
        event_no_column_receipt_list = receipt_pa['event_no'].to_pylist()

        if not event_no_column_truth_list or not event_no_column_receipt_list:
            return pa.Table.from_pydict({field.name: [] for field in self._MERGED_SCHEMA}, schema=self._MERGED_SCHEMA)

        lookup_options = SetLookupOptions(value_set=pa.array(event_no_column_receipt_list))
        filtered_rows = pc.is_in(pa.array(event_no_column_truth_list), options=lookup_options)
        return table.filter(filtered_rows)


    # --------- TABLE SHARD GETTERS ---------
    def _get_pa_shard(self, 
                    receipt_pa: pa.Table, 
                    event_no_subset: List[int], 
                    schema_name: str, 
                    event_no_column: str, 
                    build_query_func: callable) -> pa.Table:
        """
        Generic function to retrieve a PyArrow table shard from the database.

        Parameters:
            - receipt_pa: PyArrow Table containing receipt data.
            - event_no_subset: List of event numbers to filter by.
            - schema_name: The name of the schema (e.g., 'TRUTH', 'GNLabel').
            - event_no_column: The event number column name in the resulting table.
            - build_query_func: The query-building function specific to the schema.

        Returns:
            - A filtered PyArrow table shard.
        """
        query = build_query_func(event_no_subset)
        rows, columns = self._execute_query(query)
        
        schema = getattr(self, f"_{schema_name}_SCHEMA")
        nan_replacement = getattr(self, f"_nan_replacement_{schema_name}")

        # special case for TRUTH, else use generic builder
        table = (
            self._create_truth_pa_table(rows, columns)
            if schema_name == "TRUTH"
            else self._create_trailing_pa_table(rows, columns, schema, nan_replacement)
        )
        
        return self._filter_rows(table, receipt_pa, event_no_column)


    # --------- TABLE BUILDERS ---------
    def _create_truth_pa_table(self, rows: List[tuple], columns: List[str]) -> pa.Table:
        if not rows:
            return pa.Table.from_pydict({field.name: [] for field in self._TRUTH_SCHEMA}, schema=self._TRUTH_SCHEMA)

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


    # --------- SCHEMA BUILDERS ---------
    def _build_schema(self) -> None: 
        """
        IS_DEBUG is a static constant when whose value is True, 
        the merged schema will include the event_no from GNHighestEDaughter.
        """
        IS_DEBUG = False
        
        self._TRUTH_SCHEMA = TruthEntity.get_schema('TRUTH')
        self._GNLabel_SCHEMA = TruthEntity.get_schema('GNLabel')
        self._HighestEInIceParticle_SCHEMA = TruthEntity.get_schema('HighestEInIceParticle')
        self._HE_DAUGHTER_SCHEMA = TruthEntity.get_schema('HE_DAUGHTER')
        self._MCWeightDict_SCHEMA = TruthEntity.get_schema('MCWeightDict')
        
        if IS_DEBUG:
            self._MERGED_SCHEMA = pa.schema(
                list(self._TRUTH_SCHEMA)
                + [field for field in self._GNLabel_SCHEMA]
                + [field for field in self._HighestEInIceParticle_SCHEMA]
                + [field for field in self._HE_DAUGHTER_SCHEMA]
                + [field for field in self._MCWeightDict_SCHEMA]
            )
        else:
            self._MERGED_SCHEMA = pa.schema(
                list(self._TRUTH_SCHEMA)  # FIX: Added self._TRUTH_SCHEMA here
                + [field for field in self._GNLabel_SCHEMA if field.name != 'GNLabel_event_no']
                + [field for field in self._HighestEInIceParticle_SCHEMA if field.name != 'HighestEInIceParticle_event_no']
                + [field for field in self._HE_DAUGHTER_SCHEMA if field.name != 'HE_daughter_event_no']
                + [field for field in self._MCWeightDict_SCHEMA if field.name != 'MCWeightDict_event_no']
            )


    # --------- NAN REPLACEMENTS ---------
    def _build_nan_replacement(self) -> None:
        self._nan_replacement_TRUTH = {} # assumption: no NaNs in TRUTH
        self._nan_replacement_GNLabel = TruthEntity.get_nan_replacements('GNLabel')
        self._nan_replacement_HighestEInIceParticle = TruthEntity.get_nan_replacements('HighestEInIceParticle')
        self._nan_replacement_HE_DAUGHTER = TruthEntity.get_nan_replacements('HE_DAUGHTER')
        self._nan_replacement_MCWeightDict = TruthEntity.get_nan_replacements('MCWeightDict')


    # --------- QUERY BUILDERS ---------
    def _build_truth_query(self, event_no_subset: List[int]) -> str:
        """
        Builds the SQL query for the TRUTH table, including the calculation of N_doms.
        """
        excluded_columns = TruthEntity.get_excluded_columns("TRUTH")

        # Extract the column names from the schema, excluding unwanted ones
        columns = [
            field.name for field in self._TRUTH_SCHEMA 
            if field.name not in excluded_columns and field.name != 'event_no'
        ]

        select_columns = ['t.event_no'] + [f"t.{col}" for col in columns]
        event_filter = ','.join(map(str, event_no_subset))

        return f"""
            SELECT 
                {', '.join(select_columns)},
                COUNT(DISTINCT s.string || '-' || s.dom_number) AS N_doms
            FROM {self.truth_table_name} t
            JOIN {self.source_table} s ON t.event_no = s.event_no
            WHERE t.event_no IN ({event_filter})
            GROUP BY t.event_no
        """
        
        
    def _build_trailing_query(self, schema_name: str, table_name: str, event_no_subset: List[int], alias_prefix: str = None) -> str:
        """
        Generic SQL query builder for all schemas except TRUTH.
        """
        event_filter = ','.join(map(str, event_no_subset))
        schema = getattr(self, f"_{schema_name}_SCHEMA")
        columns = [field.name for field in schema]

        event_no_column = "event_no"
        if alias_prefix:
            event_no_column = f"event_no AS {alias_prefix}"

        select_columns = [event_no_column if col.endswith("event_no") else col for col in columns]

        return f"""
            SELECT {', '.join(select_columns)}
            FROM {table_name}
            WHERE event_no IN ({event_filter})
        """


    def _build_GNLabel_query(self, event_no_subset: List[int]) -> str:
        return self._build_trailing_query('GNLabel', self.truth_table_name, event_no_subset, alias_prefix='GNLabel_event_no')


    def _build_HighestEInIceParticle_query(self, event_no_subset: List[int]) -> str:
        return self._build_trailing_query('HighestEInIceParticle', self.HighestEInIceParticle_table_name, event_no_subset, alias_prefix='HighestEInIceParticle_event_no')


    def _build_HE_daughter_query(self, event_no_subset: List[int]) -> str:
        return self._build_trailing_query('HE_DAUGHTER', self.HE_dauther_table_name, event_no_subset, alias_prefix='HE_daughter_event_no')


    def _build_MCWeightDict_query(self, event_no_subset: List[int]) -> str:
        return self._build_trailing_query('MCWeightDict', self.MC_weight_dict_table_name, event_no_subset, alias_prefix='MCWeightDict_event_no')


    def _execute_query(self, query: str) -> (List[tuple], List[str]):
        cursor = self.con_source.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return rows, columns