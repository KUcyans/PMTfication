import pyarrow as pa
import numpy as np
from matplotlib.path import Path
import pyarrow.compute as pc
from typing import List
from pyarrow.compute import SetLookupOptions

class PMTTruthFromTruth:
    '''
    based on truth columns, 
    make a new table for the truth table
    the returned table will be joined with the truth table in TruthMaker
    '''
    def __init__(self, pa_truth: pa.Table) -> None:
        self.pa_truth = pa_truth
    
    def __call__(self) -> pa.Table:
        return self._build_truth_sub_pa()
    
    def _build_truth_sub_pa(self) -> pa.Table:
        """
        Combines the output of all processing functions into a single PyArrow Table.
        """
        processing_funcs = [
            self._is_within_IceCube,
            # add future processing functions here
        ]

        # collect results
        tables = [func() for func in processing_funcs]

        merged_table = tables[0]
        for table in tables[1:]:
            merged_table = merged_table.join(table, keys=['event_no'], join_type='inner')

        return merged_table
    
    # a processing function
    def _is_within_IceCube(self) -> pa.Table:
        border_xy = np.array([(-256.14, -521.08), (-132.80, -501.45), (-9.13, -481.74), 
                                   (114.39, -461.99), (237.78, -442.42), (361.0, -422.83), 
                                   (405.83, -306.38), (443.60, -194.16), (500.43, -58.45), 
                                   (544.07, 55.89), (576.37, 170.92), (505.27, 257.88), 
                                   (429.76, 351.02), (338.44, 463.72), (224.58, 432.35), 
                                   (101.04, 412.79), (22.11, 509.5), (-101.06, 490.22), 
                                   (-224.09, 470.86), (-347.88, 451.52), (-392.38, 334.24), 
                                   (-437.04, 217.80), (-481.60, 101.39), (-526.63, -15.60), 
                                   (-570.90, -125.14), (-492.43, -230.16), (-413.46, -327.27), 
                                   (-334.80, -424.5)])
        border_z = np.array([-512.82, 524.56])
        xy_path = Path(border_xy)
        event_no = self.pa_truth.column("event_no").to_numpy()
        pos_x = self.pa_truth.column("pos_x_GNHighestEDaughter").to_numpy()
        pos_y = self.pa_truth.column("pos_y_GNHighestEDaughter").to_numpy()
        pos_z = self.pa_truth.column("pos_z_GNHighestEDaughter").to_numpy()

        xy_points = np.column_stack((pos_x, pos_y))
        xy_mask = xy_path.contains_points(xy_points)

        z_mask = (pos_z >= border_z[0]) & (pos_z <= border_z[1])

        is_contained = xy_mask & z_mask
        
        containment_table = pa.Table.from_arrays(
            [pa.array(event_no, type=pa.int32()),
                pa.array(is_contained, type=pa.int32()),],
            names=['event_no', 'isWithinIceCube'],)
        return containment_table
    