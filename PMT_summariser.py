import pyarrow as pa
import numpy as np
from collections import defaultdict
from typing import List
import sqlite3 as sql

class PMTSummariser:
    def __init__(self, con_source: sql.Connection, source_table: str, event_no_subset: List[int]) -> None:
        self.con_source = con_source
        self.source_table = source_table
        self.event_no_subset = event_no_subset

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
            for doms_pulses in strings_doms_pulses.values():
                all_pulses_event = list(doms_pulses.values())
                maxQtotal = self._get_max_Q_total(all_pulses_event)
                avg_dom_position = self._get_Q_weighted_average_DOM_position(all_pulses_event, maxQtotal)
                for pulses in doms_pulses.values():
                    dom_data = self._process_DOM(pulses, avg_dom_position)
                    processed_data.append(dom_data)

        # Define PyArrow schema
        schema = pa.schema([
            # Define column names and their data types
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

        # Convert processed_data into PyArrow arrays
        # columns_data = list(zip(*processed_data))  # Transpose to match columnar structure
        # arrays = [pa.array(col, type=schema.field(i).type) for i, col in enumerate(columns_data)]
        # pa_processed = pa.Table.from_arrays(arrays, schema=schema)
        # return pa_processed
        
        arrays = {
            "dom_x": [],
            "dom_y": [],
            "dom_z": [],
            "dom_x_rel": [],
            "dom_y_rel": [],
            "dom_z_rel": [],
            "pmt_area": [],
            "rde": [],
            "saturation_status": [],
            "q1": [],
            "q2": [],
            "q3": [],
            "Q25": [],
            "Q75": [],
            "Qtotal": [],
            "hlc1": [],
            "hlc2": [],
            "hlc3": [],
            "t1": [],
            "t2": [],
            "t3": [],
            "T10": [],
            "T50": [],
            "sigmaT": []
        }
                
        for dom_data in processed_data:
            arrays["dom_x"].append(dom_data[0])
            arrays["dom_y"].append(dom_data[1])
            arrays["dom_z"].append(dom_data[2])
            arrays["dom_x_rel"].append(dom_data[3])
            arrays["dom_y_rel"].append(dom_data[4])
            arrays["dom_z_rel"].append(dom_data[5])
            arrays["pmt_area"].append(dom_data[6])
            arrays["rde"].append(dom_data[7])
            arrays["saturation_status"].append(dom_data[8])
            arrays["q1"].append(dom_data[9])
            arrays["q2"].append(dom_data[10])
            arrays["q3"].append(dom_data[11])
            arrays["Q25"].append(dom_data[12])
            arrays["Q75"].append(dom_data[13])
            arrays["Qtotal"].append(dom_data[14])
            arrays["hlc1"].append(dom_data[15])
            arrays["hlc2"].append(dom_data[16])
            arrays["hlc3"].append(dom_data[17])
            arrays["t1"].append(dom_data[18])
            arrays["t2"].append(dom_data[19])
            arrays["t3"].append(dom_data[20])
            arrays["T10"].append(dom_data[21])
            arrays["T50"].append(dom_data[22])
            arrays["sigmaT"].append(dom_data[23])
        # instead of using zip(*processed_data), we can directly append the values
        # for dom_data in processed_data:
        #     for i, key in enumerate(arrays.keys()):
        #         arrays[key].append(dom_data[i])
        pa_arrays = {key: pa.array(value) for key, value in arrays.items()}
        return pa.Table.from_pydict(pa_arrays)
    
    def _build_events_doms_pulses(self, rows: List[List[float]]) -> defaultdict:
        events_doms_pulses = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for row in rows:
            event_no = row[self.event_no_idx]
            string = row[self.dom_string_idx]
            dom_number = row[self.dom_number_idx]
            events_doms_pulses[event_no][string][dom_number].append(row)
        return events_doms_pulses

    def _process_DOM(self, 
                    pulses: List[List[float]], 
                    avg_dom_position: List[float]) -> List[float]:
        # Get DOM features
        # dom_string = self._get_DOM_string(pulses)
        # dom_number = self._get_DOM_number(pulses)
        dom_x, dom_y, dom_z = self._get_DOM_position(pulses)
        dom_x_rel, dom_y_rel, dom_z_rel = self._get_relative_DOM_position(dom_x, dom_y, dom_z, avg_dom_position)
        pmt_area = self._get_pmt_area(pulses)
        rde = self._get_rde(pulses)
        saturation_status = self._get_saturation_status(pulses)
        
        # Get remaining features
        first_three_charge_readout = self._get_first_charge_readout(pulses, saturation_status)
        accumulated_charge_after_nano_sec = self._get_accumulated_charge_after_ns(pulses, saturation_status)
        first_three_pulse_time = self._get_first_pulse_time(pulses, saturation_status)
        # first_three_hlc_pulse_time = self._get_first_hlc_pulse_time(pulses, saturation_status)
        first_three_hlc = self._get_first_hlc(pulses)
        elapsed_time_until_charge_fraction = self._get_elapsed_time_until_charge_fraction(pulses, saturation_status)
        standard_deviation = self._get_time_standard_deviation(first_three_pulse_time, saturation_status)
        
        data_dom = (
                    # [dom_string, dom_number]            # dom_number
                    [dom_x, dom_y, dom_z]             # dom_x, dom_y, dom_z
                    + [dom_x_rel, dom_y_rel, dom_z_rel] # dom_x_rel, dom_y_rel, dom_z_rel
                    + [pmt_area, rde, saturation_status]# pmt_area, rde, saturationStatus
                    + first_three_charge_readout        # q1, q2, q3
                    + accumulated_charge_after_nano_sec # Q25, Q75, Qtotal
                    + first_three_hlc                   # hlc1, hlc2, hlc3
                    + first_three_pulse_time            # t1, t2, t3
                    # + first_three_hlc_pulse_time        # t1_hlc, t2_hlc, t3_hlc
                    + elapsed_time_until_charge_fraction# T10, T50
                    + [standard_deviation]              # sigmaT
                    )
        return data_dom
    
    def _get_max_Q_total(self,
                        all_pulses_event: List[List[List[float]]]) -> float:
                charges = np.array([[pulse[self.dom_charge_idx] for pulse in pulses] for pulses in all_pulses_event])
                Qsums = np.sum(charges, axis=1)
                return np.max(Qsums)
            
    def _get_Q_weighted_average_DOM_position(self, 
                                        all_pulses_event: List[List[List[float]]], 
                                        maxQtotal: float) -> List[float]:
        dom_positions = np.array([
            [pulse[self.dom_x_idx], pulse[self.dom_y_idx], pulse[self.dom_z_idx]]
            for pulses_dom in all_pulses_event for pulse in pulses_dom
        ])
        charges = np.array([
            pulse[self.dom_charge_idx] for pulses_dom in all_pulses_event for pulse in pulses_dom
        ])

        # Calculate weighted averages
        weights = charges / maxQtotal
        weighted_positions = np.average(dom_positions, axis=0, weights=weights)
        weighted_x, weighted_y, weighted_z = weighted_positions

        return [weighted_x, weighted_y, weighted_z]
        
    def _get_relative_DOM_position(self,
                                dom_x: float, dom_y: float, dom_z: float, 
                                avg_dom_position: List[float]) -> List[float]:
        return [dom_x - avg_dom_position[0], dom_y - avg_dom_position[1], dom_z - avg_dom_position[2]]
    
    # NOTE pulses_dom: [pulse, ...]
    def _get_DOM_position(self,
                        pulses_dom: List[List[float]],) -> List[float]:
        return [pulses_dom[0][self.dom_x_idx], pulses_dom[0][self.dom_y_idx], pulses_dom[0][self.dom_z_idx]]
    
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
                    pulses_dom: List[List[float]]) -> List[int]:
        n = 3
        _fillIncomplete = -1
        if len(pulses_dom) < n:
            hlc = [pulse[self.dom_hlc_idx] for pulse in pulses_dom]
            hlc.extend([_fillIncomplete] * (n - len(hlc)))
        else:
            hlc = [pulse[self.dom_hlc_idx] for pulse in pulses_dom[:n]]
        return hlc
    
    def _get_first_pulse_time(self, 
                            pulses_dom: List[List[float]], 
                            saturationStatus: int) -> List[float]:
        n = 3
        # HACK consider changing the fill values
        _fillSaturated = -1
        _fillIncomplete = -1
        
        if saturationStatus == 1:
            pulse_times = [_fillSaturated] * n
        elif len(pulses_dom) < n:
            pulse_times = [pulse[self.dom_time_idx] for pulse in pulses_dom]
            pulse_times.extend([_fillIncomplete] * (n - len(pulse_times)))
        else:
            pulse_times = [pulse[self.dom_time_idx] for pulse in pulses_dom[:n]]
        return pulse_times
    
    # HACK necessary?
    def _get_first_hlc_pulse_time(self,
                                pulses_dom: List[List[float]], 
                                saturationStatus: int) -> List[float]:
        n = 3
        _fillSaturated = -1 # used when the DOM is saturated
        _fillIncomplete = -1 # Used there 
        if saturationStatus == 1:
            pulse_times = [_fillSaturated] * n
        elif len(pulses_dom) < n:
            pulse_times = [pulse[self.dom_time_idx] for pulse in pulses_dom if pulse[self.dom_hlc_idx] == 1]
            pulse_times.extend([_fillIncomplete] * (n - len(pulse_times)))
        else:
            pulse_times = [pulse[self.dom_time_idx] for pulse in pulses_dom if pulse[self.dom_hlc_idx] == 1][:n]
        return pulse_times
        
    def _get_elapsed_time_until_charge_fraction(self,
                                                pulses_dom: List[List[float]], 
                                                saturationStatus: int, 
                                                percentile1 = 10, 
                                                percentile2 = 50,
                                                ) -> List[float]:
        # HACK consider changing the fill values
        _fillSaturated = -1
        _fillIncomplete = -1
        if saturationStatus == 1:
            times = [_fillSaturated] * 2
        elif len(pulses_dom) < 2:
            times = [_fillIncomplete] * 2
        else:
            charges = np.array([pulse[self.dom_charge_idx] for pulse in pulses_dom])
            times = np.array([pulse[self.dom_time_idx] for pulse in pulses_dom])
            Qtotal = np.sum(charges)
            cumulated_charge = np.cumsum(charges)
            t_0 = times[0]
            T_1 = times[np.argmax(cumulated_charge > percentile1 / 100 * Qtotal)] - t_0
            T_2 = times[np.argmax(cumulated_charge > percentile2 / 100 * Qtotal)] - t_0
            
            # Qtotal = sum([pulse[self.dom_charge_idx] for pulse in pulses_dom])
            # t_0 = pulses_dom[0][self.dom_time_idx]
            # Qcum = 0
            # T_first, T_second = -1, -1 # if these are not -1, then they are assigned
            # for pulse in pulses_dom:
            #     Qcum += pulse[self.dom_charge_idx]
            #     if Qcum > percentile1 / 100 * Qtotal and T_first == -1:
            #         T_first = pulse[self.dom_time_idx] - t_0
            #     if Qcum > percentile2 / 100 * Qtotal:
            #         T_second = pulse[self.dom_time_idx] - t_0
            #         break
            times = [T_1, T_2]
        return times
    
    def _get_time_standard_deviation(self,
                                    pulse_times: List[float], 
                                    saturationStatus: int) -> float:
        # HACK consider changing the fill values
        _fillSaturated = 0
        _fillIncomplete = 0
        if saturationStatus == 1:
            sigmaT = _fillSaturated
        elif len(pulse_times) < 2:
            sigmaT = _fillIncomplete
        else:
            sigmaT = np.std(pulse_times)
        return sigmaT
    
    def _get_first_charge_readout(self,
                                pulses: List[List[float]], 
                                saturationStatus: int) -> List[float]:
        # HACK consider changing the fill values
        _fillSaturated = -1
        _fillIncomplete = -1
        n = 3
        if saturationStatus == 1:
            charge_readouts = [_fillSaturated] * n
        elif len(pulses) < n:
            charge_readouts = [pulse[self.dom_charge_idx] for pulse in pulses]
            charge_readouts.extend([_fillIncomplete] * (n - len(charge_readouts)))
        else:
            charge_readouts = [pulse[self.dom_charge_idx] for pulse in pulses[:n]]
        return charge_readouts
    
    def _get_accumulated_charge_after_ns(self,
                                        pulses: List[List[float]], 
                                        saturationStatus: int, 
                                        interval1 = 25, 
                                        interval2 = 75) -> List[float]:
        # HACK consider changing the fill values
        _fillSaturated = -1
        _fillIncomplete = -1
        if saturationStatus == 1:
            Qs = [_fillSaturated] * 3
        elif len(pulses) < 1:
            Qs = [_fillIncomplete] * 3
        else:
            t_0 = pulses[0][self.dom_time_idx]
            times = np.array([pulse[self.dom_time_idx] for pulse in pulses])
            charges = np.array([pulse[self.dom_charge_idx] for pulse in pulses])

            # Calculate cumulative charge within intervals
            time_offsets = times - t_0
            Qinterval1 = np.sum(charges[time_offsets < interval1])
            Qinterval2 = np.sum(charges[time_offsets < interval2])
            Qtotal = np.sum(charges)
            Qs = [Qinterval1, Qinterval2, Qtotal]
        return Qs