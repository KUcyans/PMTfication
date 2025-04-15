import pyarrow as pa
import numpy as np
from matplotlib.path import Path
import pyarrow.compute as pc
from typing import List
from pyarrow.compute import SetLookupOptions
from functools import lru_cache
from typing import Union

class PMTTruthFromTruth:
    '''
    based on truth columns, 
    make a new table for the truth table
    the returned table will be joined with the truth table in TruthMaker
    '''
    _PRISM_FACES: List[np.ndarray] = None  # class-level cache

    def __init__(self, pa_truth: pa.Table) -> None:
        self.pa_truth = pa_truth
        if PMTTruthFromTruth._PRISM_FACES is None:
            PMTTruthFromTruth._PRISM_FACES = self._build_icecube_prism_faces()
    
    def __call__(self) -> pa.Table:
        return self._build_truth_sub_pa()
    
    def _build_truth_sub_pa(self) -> pa.Table:
        """
        Combines the output of all processing functions into a single PyArrow Table.
        """
        processing_funcs = [
            self._is_within_IceCube,
            self._compute_intra_IceCube_lepton_travel_distance,
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
    
    
    # ----- Intra IceCube lepton travel distance calculation -----
    def _compute_intra_IceCube_lepton_travel_distance(self) -> pa.Table:
        """
        Computes the travel distance of each event's HE daughter line inside a fixed prism.
        Returns a PyArrow table with event_no and travel_distance_inside_volume.
        """
        # Define your prism faces (list of corner sets; each face is a polygon in 3D)
        # For this example, I’ll assume you have a constant like this defined elsewhere
        # prism_faces = self._get_prism_faces()

        # Load vectors from the table
        event_no = self.pa_truth.column("event_no").to_numpy()
        pos = np.stack([
            self.pa_truth.column("pos_x_GNHighestEDaughter").to_numpy(),
            self.pa_truth.column("pos_y_GNHighestEDaughter").to_numpy(),
            self.pa_truth.column("pos_z_GNHighestEDaughter").to_numpy()
        ], axis=1)

        dir_vec = np.stack([
            self.pa_truth.column("dir_x_GNHighestEDaughter").to_numpy(),
            self.pa_truth.column("dir_y_GNHighestEDaughter").to_numpy(),
            self.pa_truth.column("dir_z_GNHighestEDaughter").to_numpy()
        ], axis=1)

        # Compute distances
        distances = []
        for p, d in zip(pos, dir_vec):
            intersections = []
            for face in PMTTruthFromTruth._PRISM_FACES:
                point = self._compute_intersection_with_plane(p, d, face)
                if point is not None and self._check_intersection_containment(point, face):
                    intersections.append(point)

            # Deduplicate nearby points
            unique_points = []
            for pt in intersections:
                if not any(np.linalg.norm(pt - prev_pt) < 1e-6 for prev_pt in unique_points):
                    unique_points.append(pt)

            # Project to line via t, sort
            v_norm_sq = np.dot(d, d)
            t_values = [(np.dot(pt - p, d) / v_norm_sq, pt) for pt in unique_points]
            t_values.sort(key=lambda x: x[0])

            if len(t_values) < 2:
                distances.append(0.0)
            else:
                (t1, pt1), (t2, pt2) = t_values
                if t1 >= 0:
                    distances.append(np.linalg.norm(pt2 - pt1))
                elif t2 > 0:
                    distances.append(np.linalg.norm(pt2 - p))
                else:
                    distances.append(0.0)

        return pa.Table.from_pydict({
            'event_no': pa.array(event_no, type=pa.int32()),
            'lepton_intra_distance': pa.array(distances, type=pa.float32())
        })

    def _build_icecube_prism_faces(self):
        ICECUBE_FACES = {
            0: np.array([(269.70961549, 548.30058428, 524.56), (269.70961549, 548.30058428, -512.82),
                            (576.36999512, 170.91999817, 524.56), (576.36999512, 170.91999817, -512.82)]),
            1: np.array([(576.36999512, 170.91999817, 524.56), (576.36999512, 170.91999817, -512.82),
                            (361.0, -422.82998657, 524.56), (361.0, -422.82998657, -512.82)]),
            2: np.array([(361.0, -422.82998657, 524.56), (361.0, -422.82998657, -512.82),
                            (-256.14001465, -521.08001709, 524.56), (-256.14001465, -521.08001709, -512.82)]),
            3: np.array([(-256.14001465, -521.08001709, 524.56), (-256.14001465, -521.08001709, -512.82),
                            (-570.90002441, -125.13999939, 524.56), (-570.90002441, -125.13999939, -512.82)]),
            4: np.array([(-570.90002441, -125.13999939, 524.56), (-570.90002441, -125.13999939, -512.82),
                            (-347.88000488, 451.51998901, 524.56), (-347.88000488, 451.51998901, -512.82)]),
            5: np.array([(-347.88000488, 451.51998901, 524.56), (-347.88000488, 451.51998901, -512.82),
                            (269.70961549, 548.30058428, 524.56), (269.70961549, 548.30058428, -512.82)]),
        }
        ICECUBE_BASE_CORNERS = {
            0: np.array([(269.70961549, 548.30058428, -512.82), 
                        (576.36999512, 170.91999817, -512.82),
                        (361.0, -422.82998657, -512.82), 
                        (-256.14001465, -521.08001709, -512.82),
                        (-570.90002441, -125.13999939, -512.82),
                        (-347.88000488, 451.51998901, -512.82)]),
            1: np.array([(269.70961549, 548.30058428, 524.56),
                        (576.36999512, 170.91999817, 524.56),
                        (361.0, -422.82998657, 524.56), 
                        (-256.14001465, -521.08001709, 524.56),
                        (-570.90002441, -125.13999939, 524.56),
                        (-347.88000488, 451.51998901, 524.56)]),
            }
        
        return list(ICECUBE_FACES.values()) + list(ICECUBE_BASE_CORNERS.values())

    def _compute_intersection_with_plane(self, pos: np.ndarray, direction: np.ndarray, corner_set: np.ndarray) -> Union[np.ndarray, None]:
        """
        Computes intersection point of a line with a plane defined by a polygon.
        
        Parameters:
            pos : np.ndarray
                Starting point of the line, shape (3,)
            direction : np.ndarray
                Direction vector of the line, shape (3,)
            corner_set : np.ndarray
                At least 3 points that define a plane, shape (N, 3)
        
        Returns:
            np.ndarray or None: The intersection point, or None if parallel to the plane.
        """
        corner_0, corner_1, corner_2 = corner_set[:3]
        edge01 = corner_1 - corner_0
        edge12 = corner_2 - corner_1
        normal = np.cross(edge01, edge12)
        normal /= np.linalg.norm(normal)

        denom = np.dot(normal, direction)
        if abs(denom) < 1e-12:  # Line is parallel to plane
            return None

        t = np.dot(normal, corner_0 - pos) / denom
        intersection = pos + t * direction
        return intersection

    
    def _check_intersection_containment(self, intersection: np.ndarray, corner_set: np.ndarray) -> bool:
        """
        Check if a 3D point lies within a polygon (3D) defined by corner_set.
        Projects to 2D and uses a soft tolerance for robust containment.
        
        Parameters:
            intersection : np.ndarray
                A 3D point (shape: (3,))
            corner_set : np.ndarray
                An array of polygon corners (shape: (N, 3)), assumed to lie in the same plane.
        
        Returns:
            bool: True if intersection lies inside the polygon, False otherwise.
        """
        # Define the plane using first three points
        p0, p1, p2 = corner_set[:3]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)

        # Orthonormal basis vectors in the plane
        u = v1 / np.linalg.norm(v1)
        w = np.cross(normal, u)

        def project_to_plane(pt):
            vec = pt - p0
            return np.dot(vec, u), np.dot(vec, w)

        # Project the corners and the intersection
        polygon_2d = np.array([project_to_plane(c) for c in corner_set])
        point_2d = project_to_plane(intersection)

        # Use 2D polygon containment check with numerical tolerance
        return Path(polygon_2d, closed=True).contains_point(point_2d, radius=1e-8)
