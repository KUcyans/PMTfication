import pyarrow as pa
import numpy as np
from collections import defaultdict
from typing import List, Dict
import sqlite3 as sql

from PMT_ref_pos_adder import ReferencePositionAdder

class PMTSummariser:
    """
    NOTE data structure
    events_doms_pulses  : {
        event_no: {
            string: {
                dom_number: [pulse, ...], 
                ...
            }, 
            ...
        }, 
        ...
    }
    """
    _SCHEMA = None
    _DEFAULT_ARRAYS = None

    def __init__(self, 
                 con_source: sql.Connection, 
                 source_table: str, 
                 event_no_subset: List[int],
                 n_pulse_collect: int = 5
                 ) -> None:
        self.con_source = con_source
        self.source_table = source_table
        self.event_no_subset = event_no_subset
        self.n_pulse_collect = n_pulse_collect

        query = f"SELECT * FROM {self.source_table} LIMIT 1"
        cur_source = self.con_source.cursor()
        cur_source.execute(query)
        columns = [description[0] for description in cur_source.description]
        need_string_dom_number = 'string' not in columns or 'dom_number' not in columns
        
        if need_string_dom_number:
            # NOTE 
            # ReferencePositionAdder adds string and dom_number based on 
            # the reference data
            # HACK The reference is hardcoded in PMT_ref_pos_adder.py
            ref_pos_adder = ReferencePositionAdder(
                con_source=con_source,
                source_table=source_table,
                event_no_subset=event_no_subset,
                tolerance_xy=10,
                tolerance_z=2
            )
            ref_pos_adder()

        self.event_no_idx = columns.index('event_no')
        self.dom_string_idx = columns.index('string')
        self.dom_number_idx = columns.index('dom_number')
        self.dom_x_idx = columns.index('dom_x')
        self.dom_y_idx = columns.index('dom_y')
        self.dom_z_idx = columns.index('dom_z')
        self.dom_time_idx = columns.index('dom_time')
        self.dom_hlc_idx = columns.index('hlc')
        self.dom_charge_idx = columns.index('charge')
        self.pmt_area_idx = columns.index('pmt_area')
        self.rde_idx = columns.index('rde')
        self.saturation_status_idx = columns.index('is_saturated_dom')
        self.bad_dom_status_idx = columns.index('is_bad_dom')
        self.bright_dom_status_idx = columns.index('is_bright_dom')

        if PMTSummariser._SCHEMA is None:
            PMTSummariser._SCHEMA = self._build_schema()
        if PMTSummariser._DEFAULT_ARRAYS is None:
            PMTSummariser._DEFAULT_ARRAYS = self._build_empty_arrays()
    
    def __call__(self) -> pa.Table:
        return self._get_PMTfied_pa()
    
    def _get_PMTfied_pa(self) -> pa.Table:
        event_filter = ','.join(map(str, self.event_no_subset))
        query = f"""SELECT * 
                    FROM {self.source_table}
                    WHERE event_no IN ({event_filter})
                """
        cur_source = self.con_source.cursor()
        cur_source.execute(query)
        rows = cur_source.fetchall()

        events_doms_pulses = self._build_events_doms_pulses(rows)
        processed_data = []

        for event_no, strings_doms_pulses in events_doms_pulses.items():
            avg_dom_position = self._get_Q_weighted_DOM_position(strings_doms_pulses)  # one per event
            for string, doms_pulses in strings_doms_pulses.items():
                for dom_no, pulses in doms_pulses.items():
                    dom_data = self._process_DOM(pulses, avg_dom_position)
                    processed_data.append(np.hstack(([event_no], dom_data)))  # Convert to NumPy row

        # Convert list of arrays into a single NumPy array (efficient bulk processing)
        processed_data = np.vstack(processed_data) if processed_data else np.empty((0, len(PMTSummariser._SCHEMA.names)), dtype=np.float32)

        # Build PyArrow Table
        pa_arrays = {field: pa.array(processed_data[:, idx]) for idx, field in enumerate(PMTSummariser._SCHEMA.names)}
        return pa.Table.from_pydict(pa_arrays, schema=PMTSummariser._SCHEMA)


    def _build_events_doms_pulses(self, rows: List[tuple]) -> defaultdict[int, defaultdict[int, defaultdict[int, np.ndarray]]]:
        """ 
        Convert the nested dictionary structure but ensure `pulses` are stored as NumPy arrays.
        """
        events_doms_pulses = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # Step 1: Populate lists
        for row in rows:
            event_no = row[self.event_no_idx]
            string = row[self.dom_string_idx]
            dom_number = row[self.dom_number_idx]
            events_doms_pulses[event_no][string][dom_number].append(row)

        # Step 2: Convert lists to NumPy arrays for fast numerical processing
        for event_no, string_dict in events_doms_pulses.items():
            for string, dom_dict in string_dict.items():
                for dom_no, pulse_list in dom_dict.items():
                    events_doms_pulses[event_no][string][dom_no] = np.array(pulse_list, dtype=np.float32)

        return events_doms_pulses


    def _get_Q_weighted_DOM_position(self, strings_doms_pulses: Dict[int, Dict[int, np.ndarray]]) -> np.ndarray:
        total_weighted_position = np.zeros(3, dtype=np.float32)
        Q_event = 0.0
        for doms_pulses in strings_doms_pulses.values():
            for pulses in doms_pulses.values():
                q = pulses[:, self.dom_charge_idx]
                total_weighted_position += np.sum(pulses[:, [self.dom_x_idx, self.dom_y_idx, self.dom_z_idx]] * q[:, None], axis=0)
                Q_event += np.sum(q)
        return total_weighted_position / Q_event if Q_event > 0 else np.zeros(3, dtype=np.float32)
    
    def _process_DOM(self, pulses: np.ndarray, avg_dom_position: np.ndarray) -> np.ndarray:
        dom_position = self._get_DOM_position(pulses)
        rel_dom_pos = self._get_relative_DOM_position(dom_position, avg_dom_position)

        pmt_area = self._get_pmt_area(pulses)
        rde = self._get_rde(pulses)
        saturation_status = self._get_saturation_status(pulses)
        bad_dom_status = self._get_bad_dom_status(pulses)
        bright_dom_status = self._get_bright_dom_status(pulses)

        first_charge_readout = self._get_first_charge_readout(pulses)
        accumulated_charge_after_nc = self._get_accumulated_charge_after_ns(pulses)
        first_pulse_time = self._get_first_pulse_time(pulses)
        first_hlc = self._get_first_hlc(pulses)
        elapsed_time_until_charge_fraction = self._get_elapsed_time_until_charge_fraction(pulses)
        standard_deviation = self._get_time_standard_deviation(pulses)

        # Using np.hstack to avoid tolist() and keep NumPy efficiency
        return np.hstack((
            dom_position, rel_dom_pos,
            [pmt_area, rde, saturation_status, bad_dom_status, bright_dom_status],
            first_charge_readout, accumulated_charge_after_nc, first_hlc,
            first_pulse_time, elapsed_time_until_charge_fraction,
            [standard_deviation]
        )).astype(np.float32)

    
    def _get_relative_DOM_position(self,
                                dom_position: np.ndarray,
                                avg_dom_position: np.ndarray) -> np.ndarray:
        rel_position = dom_position - avg_dom_position
        return rel_position
    
    def _get_DOM_position(self, pulses: np.ndarray) -> np.ndarray:
        return pulses[0, [self.dom_x_idx, self.dom_y_idx, self.dom_z_idx]]

    def _get_DOM_string(self,
                        pulses: np.ndarray) -> int:
        return pulses[0, self.dom_string_idx]
    
    def _get_DOM_number(self,
                        pulses: np.ndarray) -> int:
        return pulses[0, self.dom_number_idx]
    
    def _get_pmt_area(self,
                    pulses: np.ndarray) -> float:
        return pulses[0, self.pmt_area_idx]
    
    def _get_rde(self,
                pulses: np.ndarray) -> float:
        return pulses[0, self.rde_idx]
    
    def _get_saturation_status(self,
                            pulses: np.ndarray) -> int:
        return pulses[0, self.saturation_status_idx]
    
    def _get_bad_dom_status(self,
                        pulses: np.ndarray) -> int:
        return pulses[0, self.bad_dom_status_idx]
    
    def _get_bright_dom_status(self,
                            pulses: np.ndarray) -> int:
        return pulses[0, self.bright_dom_status_idx]
    
    def _get_first_hlc(self, pulses: np.ndarray) -> np.ndarray:
        _fillIncomplete = -1
        if pulses.shape[0] < self.n_pulse_collect:
            hlc = np.pad(pulses[:, self.dom_hlc_idx], 
                        (0, self.n_pulse_collect - pulses.shape[0]), 
                        constant_values=_fillIncomplete)
        else:
            hlc = pulses[:self.n_pulse_collect, self.dom_hlc_idx]
        return hlc.astype(np.int32)

    def _get_first_charge_readout(self, pulses: np.ndarray) -> np.ndarray:
        _fillIncomplete = -1
        if pulses.shape[0] < self.n_pulse_collect:
            charge_readouts = np.pad(
                pulses[:, self.dom_charge_idx],
                (0, self.n_pulse_collect - pulses.shape[0]), 
                constant_values=_fillIncomplete
            )
        else:
            charge_readouts = pulses[:self.n_pulse_collect, self.dom_charge_idx]

        return charge_readouts.astype(np.float32)

    def _get_accumulated_charge_after_ns(self, pulses: np.ndarray, interval1=25, interval2=75) -> np.ndarray:
        _fillIncomplete = -1
        if pulses.shape[0] < 1:
            return np.full(3, _fillIncomplete, dtype=np.float32)

        t_0 = pulses[0, self.dom_time_idx]
        time_offsets = pulses[:, self.dom_time_idx] - t_0
        charges = pulses[:, self.dom_charge_idx]

        Qinterval1 = np.sum(charges[time_offsets < interval1])
        Qinterval2 = np.sum(charges[time_offsets < interval2])
        Qtotal = np.sum(charges)

        return np.array([Qinterval1, Qinterval2, Qtotal], dtype=np.float32)

    
    def _get_first_pulse_time(self, pulses: np.ndarray) -> np.ndarray:
        _fillIncomplete = -1
        if pulses.shape[0] < self.n_pulse_collect:
            pulse_times = np.pad(pulses[:, self.dom_time_idx], 
                                (0, self.n_pulse_collect - pulses.shape[0]), 
                                constant_values=_fillIncomplete)
        else:
            pulse_times = pulses[:self.n_pulse_collect, self.dom_time_idx]
        return pulse_times.astype(np.float32)
    
    def _get_elapsed_time_until_charge_fraction(self, pulses: np.ndarray, percentile1=10, percentile2=50) -> np.ndarray:
        _fillIncomplete = -1
        if pulses.shape[0] < 2:
            return np.full(2, _fillIncomplete, dtype=np.float32)

        charges = pulses[:, self.dom_charge_idx]
        times = pulses[:, self.dom_time_idx]
        Qtotal = np.sum(charges)
        cumulated_charge = np.cumsum(charges)

        # Use np.searchsorted() to avoid index errors
        idx1 = np.searchsorted(cumulated_charge, percentile1 / 100 * Qtotal, side="right")
        idx2 = np.searchsorted(cumulated_charge, percentile2 / 100 * Qtotal, side="right")

        T_1 = times[idx1] - times[0] if idx1 < pulses.shape[0] else _fillIncomplete
        T_2 = times[idx2] - times[0] if idx2 < pulses.shape[0] else _fillIncomplete

        return np.array([T_1, T_2], dtype=np.float32)


    def _get_time_standard_deviation(self, pulses: np.ndarray) -> float:
        _fillIncomplete = -1
        return np.std(pulses[:, self.dom_time_idx]) if pulses.shape[0] > 1 else _fillIncomplete

    
    def _build_schema(self) -> pa.Schema:
        base_schema = [
            ('event_no', pa.int32()),
            ('dom_x', pa.float32()),
            ('dom_y', pa.float32()),
            ('dom_z', pa.float32()),
            ('dom_x_rel', pa.float32()),
            ('dom_y_rel', pa.float32()),
            ('dom_z_rel', pa.float32()),
            ('pmt_area', pa.float32()),
            ('rde', pa.float32()),
            ('saturation_status', pa.int32()),
            ('bad_dom_status', pa.int32()),
            ('bright_dom_status', pa.int32()),
        ]
        # q1, q2, q3 (,q4, q5, ...)
        charge_columns = [
            (f'q{i+1}', pa.float32()) for i in range(self.n_pulse_collect)
        ]
        
        accumulated_charge_columns = [
            ('Q25', pa.float32()),
            ('Q75', pa.float32()),
            ('Qtotal', pa.float32())]

        hlc_columns = [
            (f'hlc{i+1}', pa.int32()) for i in range(self.n_pulse_collect)
        ]
        # t1, t2, t3 (,t4, t5, ...)
        time_columns = [
            (f't{i+1}', pa.float32()) for i in range(self.n_pulse_collect)
        ]

        accumulated_time_columns = [
            ('T10', pa.float32()),
            ('T50', pa.float32()),
            ('sigmaT', pa.float32())
        ]

        return pa.schema(
            base_schema + 
            charge_columns + 
            accumulated_charge_columns +
            hlc_columns + 
            time_columns + 
            accumulated_time_columns
        )
    
    @classmethod
    def _build_empty_arrays(cls) -> Dict[str, List[float]]:
        if cls._SCHEMA is None:
            raise ValueError("Schema must be defined before building empty arrays.")
        return {field: [] for field in cls._SCHEMA.names}
