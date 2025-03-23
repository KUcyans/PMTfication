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
                 min_travel_distance: float = 250.0):
        self.min_travel_distance = min_travel_distance
        self.build_IceCube_faces_and_bases()
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
        required_columns = {"pos_x_GNHighestEDaughter", "pos_y_GNHighestEDaughter", "pos_z_GNHighestEDaughter", 
                            "dir_x_GNHighestEDaughter", "dir_y_GNHighestEDaughter", "dir_z_GNHighestEDaughter",
                            "isWithinIceCube", "N_doms", "event_no"}
        missing_columns = required_columns - set(truth_table.column_names)
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}. Cannot proceed with filtering.")
            return 

        pos_x = truth_table.column("pos_x_GNHighestEDaughter").to_numpy()
        pos_y = truth_table.column("pos_y_GNHighestEDaughter").to_numpy()
        pos_z = truth_table.column("pos_z_GNHighestEDaughter").to_numpy()
        
        dir_x = truth_table.column("dir_x_GNHighestEDaughter").to_numpy()
        dir_y = truth_table.column("dir_y_GNHighestEDaughter").to_numpy()
        dir_z = truth_table.column("dir_z_GNHighestEDaughter").to_numpy()
        
        vertex = np.array([pos_x, pos_y, pos_z]).T
        direction = np.array([dir_x, dir_y, dir_z]).T
        
        lines = self.get_lines_from_vertex_pos_and_dir(vertex, direction)
        intersections = self.get_line_plane_intersections(lines)
        
        event_nos = truth_table.column("event_no").to_numpy()

        # for i, (vtx, dir_vec, ints) in enumerate(zip(vertex, direction, intersections)):
        #     intra_travel_distance = self.build_intra_travel_distance(vtx, ints, dir_vec)
        #     if intra_travel_distance > self.min_travel_distance:
        #         self.valid_event_nos.add(event_nos[i])
        self.valid_event_nos = self._get_valid_event_nos_from_batch(vertex, direction, intersections, event_nos)
        self.logger.info(f"Extracted {len(self.valid_event_nos)} valid events with intra-ICECUBE lepton travel distance > {self.min_travel_distance}.")

    def build_IceCube_faces_and_bases(self):
        ICECUBE_VERTEX = {
            0: np.array([(269.70961549, 548.30058428, 524.56), (269.70961549, 548.30058428, -512.82)]),
            1: np.array([(576.36999512, 170.91999817, 524.56), (576.36999512, 170.91999817, -512.82)]),
            2: np.array([(361.00000000, -422.82998657, 524.56), (361.0000000, -422.82998657, -512.82)]),
            3: np.array([(-256.14001465, -521.08001709, 524.56), (-256.14001465, -521.08001709, -512.82)]),
            4: np.array([(-570.90002441, -125.13999939, 524.56), (-570.90002441, -125.13999939, -512.82)]),
            5: np.array([(-347.88000488, 451.51998901, 524.56), (-347.88000488, 451.51998901, -512.82)]),
        }

        planes = []
        # Create rectangular face planes
        n_faces = len(ICECUBE_VERTEX)
        for i in range(n_faces):
            point1, point2 = ICECUBE_VERTEX[i]
            point3 = ICECUBE_VERTEX[(i + 1) % n_faces][0]  # Next face's first point
            plane_func, bounds_check = self.define_plane(point1, point2, point3)
            planes.append((plane_func, bounds_check))

        # Extract hexagon vertices for bases
        hexagon_vertices = [(ICECUBE_VERTEX[i][0][0], ICECUBE_VERTEX[i][0][1]) for i in range(n_faces)]

        # Create hexagonal base planes
        for z_value in [524.56, -512.82]:  # Lid and Floor
            plane_func, _ = self.define_plane(
                np.array([hexagon_vertices[0][0], hexagon_vertices[0][1], z_value]),
                np.array([hexagon_vertices[1][0], hexagon_vertices[1][1], z_value]),
                np.array([hexagon_vertices[2][0], hexagon_vertices[2][1], z_value])
            )
            bounds_check = lambda p: self.is_inside_hexagon(p, hexagon_vertices)
            planes.append((plane_func, bounds_check))

        self.ICECUBE_FACES_AND_BASES = planes
    
    
    def is_inside_hexagon(self, point, hexagon_vertices):
        """
        Returns:
            True if inside, False otherwise.
        """
        x, y = point[:2]
        poly = Path(hexagon_vertices)  # Matplotlib Path utility
        return poly.contains_point((x, y))
    
    def _get_valid_event_nos_from_batch(self,
                                        vertex: np.ndarray,
                                        direction: np.ndarray,
                                        intersections: list,
                                        event_nos: np.ndarray) -> set:
        valid_event_nos = set()
        with tqdm(total=len(event_nos), desc="Evaluating events", unit="event") as pbar:
            for i in range(len(event_nos)):
                intra_travel_distance = self.build_intra_travel_distance(vertex[i], intersections[i], direction[i])
                if intra_travel_distance > self.min_travel_distance:
                    valid_event_nos.add(event_nos[i])
                if i % 1000 == 0 or i == len(event_nos) - 1:
                    pbar.update(1000 if i + 1000 < len(event_nos) else len(event_nos) - i)
        return valid_event_nos

    def define_line(self, point: np.ndarray, direction: np.ndarray) -> callable:
        """
        Returns a parametric function of the line: r(t) = point + t * direction
        """
        return lambda t: point + t * direction
    
    def define_plane(self, 
                     point1: np.ndarray, 
                     point2: np.ndarray, 
                     point3: np.ndarray) -> (callable, callable):
        """
        Defines a plane from 3 points and a bounds-checking function.

        Returns:
            - plane_func(r): zero when r lies on the plane. Supports single point or batch of points.
            - bounds_check(point): True if point lies within the rectangular bounds.
        """
        # Compute normal vector to the plane
        v1 = point2 - point1
        v2 = point3 - point1
        normal = np.cross(v1, v2)

        # Vectorised plane function
        def plane_func(r):
            r = np.asarray(r)
            diff = r - point1
            if diff.ndim == 1:
                return np.dot(diff, normal)  # scalar
            return diff @ normal  # shape (N,)

        # Precompute bounding box
        min_xyz = np.minimum.reduce([point1, point2, point3])
        max_xyz = np.maximum.reduce([point1, point2, point3])

        # Use member function for bounds checking
        def bounds_check(point, eps=1e-4):
            return self.is_inside_bounds(point, min_xyz, max_xyz, eps)

        return plane_func, bounds_check

    
    def is_point_on_plane(self, point: np.ndarray, plane_func, tol=1e-6) -> bool:
        return abs(plane_func(point)) < tol

    def get_lines_from_vertex_pos_and_dir(self, vertex: np.ndarray, direction: np.ndarray) -> list:
        
        lines = []
        for i, p, d in zip(range(len(vertex)), vertex, direction):
            lines.append(self.define_line(p, d))
        return lines
    
    def get_line_plane_intersections(self, lines: list) -> list:
        with ThreadPoolExecutor() as executor:
            return list(executor.map(self._get_intersections_for_line, lines))
    
    def _get_intersections_for_line(self, line):
        intersections = []
        if line is None:
            return []
        for plane_func, bounds_check in self.ICECUBE_FACES_AND_BASES:
            pts = self.intersect_line_with_plane_binary(line, plane_func, bounds_check)
            intersections.extend(pts)
        return intersections

    def is_inside_bounds(self, point, min_xyz, max_xyz, eps=1e-4):
        return np.all(point >= (min_xyz - eps)) and np.all(point <= (max_xyz + eps))

    def intersect_line_with_plane_binary(self, 
                                  line_func: callable, 
                                  plane_func: callable, 
                                  bounds_check: callable,
                                  t_range=(-1e3, 1e3), coarse_num=100):
        ts_coarse = np.linspace(*t_range, coarse_num)
        values = np.array([plane_func(line_func(t)) for t in ts_coarse])
        
        intersections = []

        # Check for coplanar case
        if np.allclose(values, 0, atol=1e-6):
            start_point = line_func(ts_coarse[0])
            end_point = line_func(ts_coarse[-1])
            if bounds_check(start_point):
                intersections.append(start_point)
            if bounds_check(end_point):
                intersections.append(end_point)
            return intersections

        sign_changes = np.where(np.diff(np.sign(values)) != 0)[0]

        for idx in sign_changes:
            t1, t2 = ts_coarse[idx], ts_coarse[idx + 1]
            point = self.binary_search_intersection(line_func, plane_func, t1, t2)
            if bounds_check(point):
                intersections.append(point)

        # Deduplicate
        tol = 1e-4
        distinct_points = []
        for pt in intersections:
            if all(np.linalg.norm(pt - other) > tol for other in distinct_points):
                distinct_points.append(pt)

        return distinct_points

    
    def binary_search_intersection(self, line_func, plane_func, t1, t2, tol=1e-6, max_iter=30):
        for _ in range(max_iter):
            tm = (t1 + t2) / 2
            vm = plane_func(line_func(tm))
            if abs(vm) < tol:
                return line_func(tm)
            if np.sign(plane_func(line_func(t1))) != np.sign(vm):
                t2 = tm
            else:
                t1 = tm
        return line_func((t1 + t2) / 2)


    
    def build_intersection_info(self, 
                               vertex: np.ndarray, 
                               intersections: list, 
                               direction: np.ndarray):
        """
        Builds entry, exit points and intersection type based on the given intersections.
        Returns:
            entry: np.ndarray
            exit: np.ndarray
            intersection_type: IntersectionType
        """
        if len(intersections) != 2:
            return None, None, IntersectionType.NO_INTERSECTION

        if np.linalg.norm(direction) < 1e-8:
            return None, None, IntersectionType.NO_INTERSECTION

        # Use unit direction vector
        unit_dir = direction / np.linalg.norm(direction)

        def project(p):
            """Signed scalar projection along direction from vertex."""
            return np.dot(p - vertex, unit_dir)

        # Projections
        t0 = project(intersections[0])
        t1 = project(intersections[1])
        t_vertex = 0.0  # vertex is the origin of projection

        # Sort points by projection value
        if t0 < t1:
            entry, exit = intersections[0], intersections[1]
            ts = [t0, t1, t_vertex]
        else:
            entry, exit = intersections[1], intersections[0]
            ts = [t1, t0, t_vertex]

        # Diagnostic logs
        if np.isclose(t0, t1, atol=1e-6):
            print("[Warning] Entry and exit are extremely close after projection.")
        if t_vertex < min(t0, t1) or t_vertex > max(t0, t1):
            print("[Warning] Vertex lies outside the entry-exit segment.")

        # Determine geometric order
        order = sorted(range(3), key=lambda i: ts[i])  # indices of [entry, exit, vertex]
        geometric_order = order.index(2)  # vertex's position
        intersection_type = IntersectionType.get_intersection_type(2, geometric_order)

        return entry, exit, intersection_type

    def build_intra_travel_distance(self, 
                                vertex: np.ndarray, 
                                intersections: list, 
                                direction: np.ndarray) -> float:
        entry, exit, intersection_type = self.build_intersection_info(vertex, intersections, direction)
        if intersection_type == IntersectionType.NO_INTERSECTION:
            intra_dist = 0.0
        elif intersection_type == IntersectionType.SINGLE_INTERSECTION:
            intra_dist = 0.0
        elif intersection_type == IntersectionType.ENTRY_EXIT_VERTEX:
            intra_dist = 0.0
        elif intersection_type == IntersectionType.ENTRY_VERTEX_EXIT:
            intra_dist = np.linalg.norm(exit - entry)
        elif intersection_type == IntersectionType.VERTEX_ENTRY_EXIT:
            intra_dist = np.linalg.norm(exit - vertex)
        return intra_dist


class IntersectionType(Enum):
    NO_INTERSECTION = (0, 0)
    SINGLE_INTERSECTION = (1, 0)  # almost impossible
    ENTRY_EXIT_VERTEX = (2, 0)    # entry point - exit point - vertex
    ENTRY_VERTEX_EXIT = (2, 1)    # entry point - vertex - exit point 
    VERTEX_ENTRY_EXIT = (2, 2)    # vertex - entry point - exit point
    
    def __init__(self, n_intersections: int, geometric_order: int):
        self._n_intersections = n_intersections
        self._geometric_order = geometric_order

    @staticmethod
    def get_intersection_type(n_intersections: int, geometric_order: int):
        for it in IntersectionType:
            if it.n_intersections == n_intersections and it.geometric_order == geometric_order:
                return it
        return IntersectionType.NO_INTERSECTION

    @property
    def n_intersections(self) -> int:
        return self._n_intersections

    @property
    def geometric_order(self) -> int:
        return self._geometric_order

    @property
    def is_lepton_length_valid(self) -> bool:
        return self.geometric_order != 0

