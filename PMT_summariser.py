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
            avg_dom_position = self._get_Q_weighted_DOM_position(strings_doms_pulses)
            for string, doms_pulses in strings_doms_pulses.items():    
                for dom_no, pulses in doms_pulses.items():
                    dom_data = self._process_DOM(pulses, avg_dom_position)
                    processed_data.append([event_no] + dom_data)
        
        # deep copy to avoid overwriting 
        arrays = {key: value[:] for key, value in PMTSummariser._DEFAULT_ARRAYS.items()}  
        for dom_data in processed_data:
            for idx, field_name in enumerate(PMTSummariser._SCHEMA.names):
                arrays[field_name].append(dom_data[idx])

        pa_arrays = {key: pa.array(value) for key, value in arrays.items()}
        return pa.Table.from_pydict(pa_arrays, schema=PMTSummariser._SCHEMA)
    
    def _build_events_doms_pulses(self, rows: List[tuple]) -> defaultdict[int, defaultdict[int, defaultdict[int, List[tuple]]]]:
        events_doms_pulses = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for row in rows:
            event_no = row[self.event_no_idx]
            string = row[self.dom_string_idx]
            dom_number = row[self.dom_number_idx]
            events_doms_pulses[event_no][string][dom_number].append(row)
        return events_doms_pulses
    
    """
    TODO consider changing the lowest level of the data structure to np.ndarray
    def _build_events_doms_pulses(self, rows: List[tuple]) -> defaultdict[int, defaultdict[int, defaultdict[int, np.ndarray]]]:
        events_doms_pulses = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: np.empty((0, len(rows[0])), dtype=float))))

        for row in rows:
            event_no = row[self.event_no_idx]
            string = row[self.dom_string_idx]
            dom_number = row[self.dom_number_idx]

            row_array = np.array(row, dtype=float)
            events_doms_pulses[event_no][string][dom_number] = np.vstack((
                events_doms_pulses[event_no][string][dom_number], 
                row_array
            ))

        return events_doms_pulses
    """
    
    def _get_Q_weighted_DOM_position(self, 
                                     strings_doms_pulses: Dict[int, 
                                                               Dict[int, 
                                                                    List[List[float]]]]) -> np.ndarray:
        total_weighted_position = np.zeros(3)
        Q_event = 0
        
        for doms_pulses in strings_doms_pulses.values():
            for pulses in doms_pulses.values():
                for pulse in pulses:
                    q = pulse[self.dom_charge_idx]
                    total_weighted_position += np.array(
                        [pulse[self.dom_x_idx], 
                        pulse[self.dom_y_idx], 
                        pulse[self.dom_z_idx]]) * q
                    Q_event += q
        
        avg_dom_position_event = total_weighted_position / Q_event
        return avg_dom_position_event
    
    def _process_DOM(self, 
                    pulses: List[List[float]], 
                    avg_dom_position: np.ndarray) -> List[float]:
        # dom_string = self._get_DOM_string(pulses)
        # dom_number = self._get_DOM_number(pulses)
        # charge = [pulse[self.dom_charge_idx] for pulse in pulses]
        dom_position = self._get_DOM_position(pulses)
        rel_dom_pos = self._get_relative_DOM_position(dom_position, avg_dom_position)
        pmt_area = self._get_pmt_area(pulses)
        rde = self._get_rde(pulses)
        saturation_status = self._get_saturation_status(pulses)
        
        first_charge_readout = self._get_first_charge_readout(pulses, saturation_status)
        accumulated_charge_after_nc = self._get_accumulated_charge_after_ns(pulses, saturation_status)
        first_pulse_time = self._get_first_pulse_time(pulses, saturation_status)
        first_hlc = self._get_first_hlc(pulses)
        elapsed_time_until_charge_fraction = self._get_elapsed_time_until_charge_fraction(pulses, saturation_status)
        standard_deviation = self._get_time_standard_deviation(pulses, saturation_status)
        
        data_dom = (
                    dom_position.tolist()                
                    + rel_dom_pos.tolist()               
                    + [pmt_area, rde, saturation_status]
                    + first_charge_readout.tolist()
                    + accumulated_charge_after_nc.tolist()
                    + first_hlc.tolist()
                    + first_pulse_time.tolist()
                    + elapsed_time_until_charge_fraction.tolist()
                    + [standard_deviation]
                    )
        return data_dom
    
    def _get_relative_DOM_position(self,
                                dom_position: np.ndarray,
                                avg_dom_position: np.ndarray) -> np.ndarray:
        rel_position = dom_position - avg_dom_position
        return rel_position
    
    def _get_DOM_position(self, pulses_dom: List[List[float]]) -> np.ndarray:
        position = np.array([
            pulses_dom[0][self.dom_x_idx],
            pulses_dom[0][self.dom_y_idx],
            pulses_dom[0][self.dom_z_idx]
        ])
        return position

    def _get_DOM_string(self,
                        pulses_dom: List[List[float]]) -> int:
        return pulses_dom[0][self.dom_string_idx]
    
    def _get_DOM_number(self,
                        pulses_dom: List[List[float]]) -> int:
        return pulses_dom[0][self.dom_number_idx]
    
    def _get_pmt_area(self,
                    pulses_dom: List[List[float]]) -> float:
        return pulses_dom[0][self.pmt_area_idx]
    
    def _get_rde(self, 
                pulses_dom: List[List[float]]) -> float:
        return pulses_dom[0][self.rde_idx]
    
    def _get_saturation_status(
                self,
                pulses_dom: List[List[float]]) -> int:
        return pulses_dom[0][self.saturation_status_idx]
    
    def _get_first_hlc(self,
                    pulses_dom: List[List[float]]) -> np.ndarray:
        
        _fillIncomplete = -1
        if len(pulses_dom) < self.n_pulse_collect:
            hlc = [pulse[self.dom_hlc_idx] for pulse in pulses_dom]
            hlc.extend([_fillIncomplete] * (self.n_pulse_collect - len(hlc)))
            hlc = np.array(hlc, dtype=int)
        else:
            hlc = np.array([pulse[self.dom_hlc_idx] for pulse in pulses_dom[:self.n_pulse_collect]], dtype=int)
        
        return hlc
    
    def _get_first_charge_readout(self,
                                pulses: List[List[float]], 
                                saturationStatus: int) -> np.ndarray:
        # HACK consider changing the fill values
        _fillSaturated = -1
        _fillIncomplete = -1

        if saturationStatus == 1:
            charge_readouts = np.full(self.n_pulse_collect, _fillSaturated, dtype=float)
        elif len(pulses) < self.n_pulse_collect:
            charge_readouts = np.array([pulse[self.dom_charge_idx] for pulse in pulses], dtype=float)
            charge_readouts = np.pad(charge_readouts, (0, self.n_pulse_collect - len(charge_readouts)), constant_values=_fillIncomplete)
        else:
            charge_readouts = np.array([pulse[self.dom_charge_idx] for pulse in pulses[:self.n_pulse_collect]], dtype=float)

        return charge_readouts
    
    def _get_accumulated_charge_after_ns(self,
                                        pulses: List[List[float]], 
                                        saturationStatus: int, 
                                        interval1=25, 
                                        interval2=75) -> np.ndarray:
        # HACK consider changing the fill values
        _fillSaturated = -1
        _fillIncomplete = -1

        if saturationStatus == 1:
            accumulated_charges = np.full(3, _fillSaturated, dtype=float)
        elif not pulses or len(pulses) < 1:
            accumulated_charges = np.full(3, _fillIncomplete, dtype=float)
        else:
            t_0 = pulses[0][self.dom_time_idx]
            times = np.array([pulse[self.dom_time_idx] for pulse in pulses], dtype=float)
            charges = np.array([pulse[self.dom_charge_idx] for pulse in pulses], dtype=float)
            
            time_offsets = times - t_0
            Qinterval1 = np.sum(charges[time_offsets < interval1])
            Qinterval2 = np.sum(charges[time_offsets < interval2])
            Qtotal = np.sum(charges)

            accumulated_charges = np.array([Qinterval1, Qinterval2, Qtotal], dtype=float)

        return accumulated_charges
    
    def _get_first_pulse_time(self, 
                            pulses_dom: List[List[float]], 
                            saturationStatus: int) -> np.ndarray:
        _fillSaturated = -1
        _fillIncomplete = -1

        if saturationStatus == 1:
            pulse_times = np.full(self.n_pulse_collect, _fillSaturated, dtype=float)
        elif len(pulses_dom) < self.n_pulse_collect:
            pulse_times = np.array([pulse[self.dom_time_idx] for pulse in pulses_dom])
            pulse_times = np.pad(pulse_times, (0, self.n_pulse_collect - len(pulse_times)), constant_values=_fillIncomplete)
        else:
            pulse_times = np.array([pulse[self.dom_time_idx] for pulse in pulses_dom[:self.n_pulse_collect]], dtype=float)

        return pulse_times
        
    def _get_elapsed_time_until_charge_fraction(self,
                                                pulses_dom: List[List[float]], 
                                                saturationStatus: int, 
                                                percentile1 = 10, 
                                                percentile2 = 50,
                                                ) -> np.ndarray:
        # HACK consider changing the fill values
        _fillSaturated = -1
        _fillIncomplete = -1

        if saturationStatus == 1:
            times = np.full(2, _fillSaturated, dtype=float)
        elif len(pulses_dom) < 2:
            times = np.full(2, _fillIncomplete, dtype=float)
        else:
            charges = np.array([pulse[self.dom_charge_idx] for pulse in pulses_dom], dtype=float)
            times = np.array([pulse[self.dom_time_idx] for pulse in pulses_dom], dtype=float)
            Qtotal = np.sum(charges)
            cumulated_charge = np.cumsum(charges)
            t_0 = times[0]
            T_1 = times[np.argmax(cumulated_charge > percentile1 / 100 * Qtotal)] - t_0
            T_2 = times[np.argmax(cumulated_charge > percentile2 / 100 * Qtotal)] - t_0
            times = np.array([T_1, T_2], dtype=float)

        return times
    
    def _get_time_standard_deviation(self,
                                    pulses_dom: List[List[float]], 
                                    saturationStatus: int) -> float:
        # HACK consider changing the fill values
        _fillSaturated = -1
        _fillIncomplete = -1
        
        pulse_times = np.array([pulse[self.dom_time_idx] for pulse in pulses_dom])
        if saturationStatus == 1:
            sigmaT = _fillSaturated
        elif len(pulse_times) < 2:
            sigmaT = _fillIncomplete
        else:
            sigmaT = np.std(pulse_times)
        return sigmaT
    
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
        return {field.name: [] for field in cls._SCHEMA}