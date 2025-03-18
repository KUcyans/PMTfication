import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
import numpy as np
from matplotlib.path import Path

from EventFilter import EventFilter, override

class ContainmentFilter(EventFilter):
    def __init__(self, 
                 source_subdir: str, 
                 output_subdir: str, 
                 subdir_no: int, 
                 part_no: int):
        self.border_xy = np.array([(-256.14, -521.08), (-132.80, -501.45), (-9.13, -481.74), 
                                   (114.39, -461.99), (237.78, -442.42), (361.0, -422.83), 
                                   (405.83, -306.38), (443.60, -194.16), (500.43, -58.45), 
                                   (544.07, 55.89), (576.37, 170.92), (505.27, 257.88), 
                                   (429.76, 351.02), (338.44, 463.72), (224.58, 432.35), 
                                   (101.04, 412.79), (22.11, 509.5), (-101.06, 490.22), 
                                   (-224.09, 470.86), (-347.88, 451.52), (-392.38, 334.24), 
                                   (-437.04, 217.80), (-481.60, 101.39), (-526.63, -15.60), 
                                   (-570.90, -125.14), (-492.43, -230.16), (-413.46, -327.27), 
                                   (-334.80, -424.5)])
        self.border_z = np.array([-512.82, 524.56])
        self.xy_path = Path(self.border_xy)
        super().__init__(source_subdir=source_subdir, 
                        output_subdir=output_subdir, 
                        subdir_no=subdir_no, 
                        part_no=part_no)

    @override
    # def _set_valid_event_nos(self): 
    #     truth_table = pq.read_table(self.source_truth_file)
        
    #     required_columns = {"pos_x_GNHighestEDaughter", "pos_y_GNHighestEDaughter", 
    #                         "pos_z_GNHighestEDaughter", "N_doms", "event_no"}
    #     missing_columns = required_columns - set(truth_table.column_names)
    #     if missing_columns:
    #         self.logger.error(f"Missing required columns: {missing_columns}. Cannot proceed with filtering.")
    #         return 

    #     pos_x = truth_table.column("pos_x_GNHighestEDaughter").to_numpy()
    #     pos_y = truth_table.column("pos_y_GNHighestEDaughter").to_numpy()
    #     pos_z = truth_table.column("pos_z_GNHighestEDaughter").to_numpy()

    #     xy_points = np.column_stack((pos_x, pos_y))
    #     xy_mask = self.xy_path.contains_points(xy_points)

    #     z_mask = (pos_z >= self.border_z[0]) & (pos_z <= self.border_z[1])

    #     valid_indices = np.where(xy_mask & z_mask)[0]

    #     if len(valid_indices) == 0:
    #         self.logger.warning(f"No valid events found within containment region in {self.subdir_no}/{self.part_no}. Skipping filtering.")
    #         return 

    #     self.valid_event_nos = set(truth_table.take(valid_indices).column("event_no").to_pylist())
    #     self.logger.info(f"Extracted {len(self.valid_event_nos)} valid events within the IceCube body.")
    
    def _set_valid_event_nos(self): 
        truth_table = pq.read_table(self.source_truth_file)
        
        required_columns = {"isWithinIceCube", "N_doms", "event_no"}
        missing_columns = required_columns - set(truth_table.column_names)
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}. Cannot proceed with filtering.")
            return 

        is_within_icecube = truth_table.column("isWithinIceCube").to_numpy()
        valid_indices = np.where(is_within_icecube)[0]

        if len(valid_indices) == 0:
            self.logger.warning(f"No valid events found within containment region in {self.subdir_no}/{self.part_no}. Skipping filtering.")
            return 

        self.valid_event_nos = set(truth_table.take(valid_indices).column("event_no").to_pylist())
        self.logger.info(f"Extracted {len(self.valid_event_nos)} valid events within the IceCube body.")