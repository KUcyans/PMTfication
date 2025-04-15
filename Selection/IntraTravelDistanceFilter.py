import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
import numpy as np
from matplotlib.path import Path
from enum import Enum
from EventFilter import EventFilter, override
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class IntraTravelDistanceFilter(EventFilter):
    def __init__(self, 
                 source_subdir: str, 
                 output_subdir: str, 
                 subdir_no: int, 
                 part_no: int,
                 min_travel_distance: float = 0.0):
        self.min_travel_distance = min_travel_distance
        super().__init__(source_subdir=source_subdir, 
                        output_subdir=output_subdir, 
                        subdir_no=subdir_no, 
                        part_no=part_no)

    @override
    def _set_valid_event_nos(self): 
        '''
        self.valid_event_nos = 
        '''
        truth_table = pq.read_table(self.source_truth_file)
        required_columns = {"lepton_intra_distance", "N_doms", "event_no"}
        missing_columns = required_columns - set(truth_table.column_names)
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}. Cannot proceed with filtering.")
            return 

        intra_icecube_lepton_travel_distance = truth_table.column("lepton_intra_distance").to_numpy()
        valid_indices = np.where(intra_icecube_lepton_travel_distance > self.min_travel_distance)[0]
        if len(valid_indices) == 0:
            self.logger.warning(f"No valid events found with intra-ICECUBE lepton travel distance > {self.min_travel_distance} in {self.subdir_no}/{self.part_no}. Skipping filtering.")
            return
        self.valid_event_nos = set(truth_table.take(valid_indices).column("event_no").to_pylist())
        self.logger.info(f"Extracted {len(self.valid_event_nos)} valid events with intra-ICECUBE lepton travel distance > {self.min_travel_distance}.")