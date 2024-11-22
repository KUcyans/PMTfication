import pyarrow as pa
import numpy as np
from collections import defaultdict
from typing import List, Dict
import sqlite3 as sql

from PMT_ref_pos_adder import ReferencePositionAdder

class PMTSummariser:
    """
NOTE data structure
events_doms_pulses  : {event_no: {string: {dom_number: [pulse, ...], ...}, ...}, ...}
strings_doms_pulses :            {string: {dom_number: [pulse, ...], ...}, ...}
doms_pulses         :                     {dom_number: [pulse, ...], ...}
pulses              :                                  [pulse, ...]
    """
    # static variables 
    _SCHEMA = None
    _DEFAULT_ARRAYS = None

    def __init__(self, con_source: sql.Connection, source_table: str, event_no_subset: List[int]) -> None:
        self.con_source = con_source
        self.source_table = source_table
        self.event_no_subset = event_no_subset
        
        # NOTE 
        # ReferencePositionAdder adds string and dom_number based on the reference data
        # HACK The reference is hardcoded in PMT_ref_pos_adder.py
        ref_pos_adder = ReferencePositionAdder(
            con_source=con_source,
            source_table=source_table,
            event_no_subset=event_no_subset,
            tolerance_xy=10,
            tolerance_z=2
        )
        ref_pos_adder()

        # Fetch column indices
        query = f"SELECT * FROM {self.source_table} LIMIT 1"
        cur_source = self.con_source.cursor()
        cur_source.execute(query)
        columns = [description[0] for description in cur_source.description]

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

        # Initialise static members if not already set
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
                                    strings_doms_pulses: Dict[int, Dict[int, List[List[float]]]]) -> np.ndarray :
        # strings_doms_pulses : {string: {dom_number: [pulse, ...], ...}, ...}
        dom_positions = []
        Qs = []
        
        for string_j, doms_pulses in strings_doms_pulses.items():
            for dom_number_i, pulses in doms_pulses.items():
                Q_ij = np.sum([pulse[self.dom_charge_idx] for pulse in pulses])
                dom_positions.append([
                    pulses[0][self.dom_x_idx], 
                    pulses[0][self.dom_y_idx], 
                    pulses[0][self.dom_z_idx]])
                Qs.append(Q_ij)
        max_Q = np.max(Qs) if Qs else 1
        Qs = np.array(Qs)
        weights = Qs / max_Q
        dom_positions = np.array(dom_positions)
        weighted_positions = np.average(dom_positions, weights=weights, axis=0)
        return weighted_positions
    
    def _process_DOM(self, 
                    pulses: List[List[float]], 
                    avg_dom_position: np.ndarray) -> List[float]:
        # Get DOM features
        # dom_string = self._get_DOM_string(pulses)
        # dom_number = self._get_DOM_number(pulses)
        # charge = [pulse[self.dom_charge_idx] for pulse in pulses]
        dom_position = self._get_DOM_position(pulses)
        rel_dom_pos = self._get_relative_DOM_position(dom_position, avg_dom_position)
        pmt_area = self._get_pmt_area(pulses)
        rde = self._get_rde(pulses)
        saturation_status = self._get_saturation_status(pulses)
        
        # Get remaining features
        first_three_charge_readout = self._get_first_charge_readout(pulses, saturation_status)
        accumulated_charge_after_nc = self._get_accumulated_charge_after_ns(pulses, saturation_status)
        first_three_pulse_time = self._get_first_pulse_time(pulses, saturation_status)
        # first_three_hlc_pulse_time = self._get_first_hlc_pulse_time(pulses, saturation_status)
        first_three_hlc = self._get_first_hlc(pulses)
        elapsed_time_until_charge_fraction = self._get_elapsed_time_until_charge_fraction(pulses, saturation_status)
        standard_deviation = self._get_time_standard_deviation(first_three_pulse_time, saturation_status)
        
        data_dom = (
                    # [dom_string, dom_number]           # dom_number
                    dom_position.tolist()                # dom_x, dom_y, dom_z
                    + rel_dom_pos.tolist()               # dom_x_rel, dom_y_rel, dom_z_rel
                    + [pmt_area, rde, saturation_status] # pmt_area, rde, saturationStatus
                    + first_three_charge_readout.tolist()# q1, q2, q3
                    + accumulated_charge_after_nc.tolist() # Q25, Q75, Qtotal
                    + first_three_hlc.tolist()          # hlc1, hlc2, hlc3
                    + first_three_pulse_time.tolist()   # t1, t2, t3
                    # + first_three_hlc_pulse_time      # t1_hlc, t2_hlc, t3_hlc
                    + elapsed_time_until_charge_fraction.tolist() # T10, T50
                    + [standard_deviation]              # sigmaT
                    )
        return data_dom
    
    def _get_relative_DOM_position(self,
                                dom_position: np.ndarray,
                                avg_dom_position: np.ndarray) -> np.ndarray:
        return dom_position - avg_dom_position
    
    # NOTE pulses_dom: [pulse, ...]
    def _get_DOM_position(self, pulses_dom: List[List[float]]) -> np.ndarray:
        return np.array([pulses_dom[0][self.dom_x_idx], pulses_dom[0][self.dom_y_idx], pulses_dom[0][self.dom_z_idx]])

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
        n = 3
        _fillIncomplete = -1

        if len(pulses_dom) < n:
            hlc = [pulse[self.dom_hlc_idx] for pulse in pulses_dom]
            hlc.extend([_fillIncomplete] * (n - len(hlc)))
            hlc = np.array(hlc, dtype=int)
        else:
            hlc = np.array([pulse[self.dom_hlc_idx] for pulse in pulses_dom[:n]], dtype=int)
        
        return hlc
    
    def _get_first_pulse_time(self, 
                            pulses_dom: List[List[float]], 
                            saturationStatus: int) -> np.ndarray:
        n = 3
        _fillSaturated = -1
        _fillIncomplete = -1

        if saturationStatus == 1:
            pulse_times = np.full(n, _fillSaturated, dtype=float)
        elif len(pulses_dom) < n:
            pulse_times = [pulse[self.dom_time_idx] for pulse in pulses_dom]
            pulse_times.extend([_fillIncomplete] * (n - len(pulse_times)))
            pulse_times = np.array(pulse_times, dtype=float)
        else:
            pulse_times = np.array([pulse[self.dom_time_idx] for pulse in pulses_dom[:n]], dtype=float)

        return pulse_times

    # HACK necessary?
    def _get_first_hlc_pulse_time(self,
                                pulses_dom: List[List[float]], 
                                saturationStatus: int) -> np.ndarray:
        n = 3
        _fillSaturated = -1
        _fillIncomplete = -1

        if saturationStatus == 1:
            pulse_times = np.full(n, _fillSaturated, dtype=float)
        elif len(pulses_dom) < n:
            pulse_times = [pulse[self.dom_time_idx] for pulse in pulses_dom if pulse[self.dom_hlc_idx] == 1]
            pulse_times.extend([_fillIncomplete] * (n - len(pulse_times)))
            pulse_times = np.array(pulse_times, dtype=float)
        else:
            pulse_times = np.array([pulse[self.dom_time_idx] for pulse in pulses_dom if pulse[self.dom_hlc_idx] == 1][:n], dtype=float)

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
                                    pulse_times: List[float], 
                                    saturationStatus: int) -> float:
        # HACK consider changing the fill values
        _fillSaturated = -1
        _fillIncomplete = -1
        if saturationStatus == 1:
            sigmaT = _fillSaturated
        elif len(pulse_times) < 2:
            sigmaT = _fillIncomplete
        else:
            sigmaT = np.std(pulse_times)
        return sigmaT
    
    def _get_first_charge_readout(self,
                                pulses: List[List[float]], 
                                saturationStatus: int) -> np.ndarray:
        # HACK consider changing the fill values
        n = 3
        _fillSaturated = -1
        _fillIncomplete = -1

        if saturationStatus == 1:
            charge_readouts = np.full(n, _fillSaturated, dtype=float)
        elif len(pulses) < n:
            charge_readouts = [pulse[self.dom_charge_idx] for pulse in pulses]
            charge_readouts.extend([_fillIncomplete] * (n - len(charge_readouts)))
            charge_readouts = np.array(charge_readouts, dtype=float)
        else:
            charge_readouts = np.array([pulse[self.dom_charge_idx] for pulse in pulses[:n]], dtype=float)

        return charge_readouts
    
    def _get_accumulated_charge_after_ns(self,
                                        pulses: List[List[float]], 
                                        saturationStatus: int, 
                                        interval1 = 25, 
                                        interval2 = 75) -> np.ndarray:
        # HACK consider changing the fill values
        _fillSaturated = -1
        _fillIncomplete = -1

        if saturationStatus == 1:
            Qs = np.full(3, _fillSaturated, dtype=float)
        elif len(pulses) < 1:
            Qs = np.full(3, _fillIncomplete, dtype=float)
        else:
            t_0 = pulses[0][self.dom_time_idx]
            times = np.array([pulse[self.dom_time_idx] for pulse in pulses], dtype=float)
            charges = np.array([pulse[self.dom_charge_idx] for pulse in pulses], dtype=float)

            # Calculate cumulative charge within intervals
            time_offsets = times - t_0
            Qinterval1 = np.sum(charges[time_offsets < interval1])
            Qinterval2 = np.sum(charges[time_offsets < interval2])
            Qtotal = np.sum(charges)
            Qs = np.array([Qinterval1, Qinterval2, Qtotal], dtype=float)

        return Qs
    
    @staticmethod
    def _build_schema() -> pa.Schema:
        """Build and return the PyArrow schema."""
        return pa.schema([
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
            ('q1', pa.float32()),
            ('q2', pa.float32()),
            ('q3', pa.float32()),
            ('Q25', pa.float32()),
            ('Q75', pa.float32()),
            ('Qtotal', pa.float32()),
            ('hlc1', pa.int32()),
            ('hlc2', pa.int32()),
            ('hlc3', pa.int32()),
            ('t1', pa.float32()),
            ('t2', pa.float32()),
            ('t3', pa.float32()),
            ('T10', pa.float32()),
            ('T50', pa.float32()),
            ('sigmaT', pa.float32())
        ])

    @classmethod
    def _build_empty_arrays(cls) -> Dict[str, List[float]]:
        """Build and return the default arrays structure."""
        if cls._SCHEMA is None:
            raise ValueError("Schema must be defined before building empty arrays.")
        return {field.name: [] for field in cls._SCHEMA}