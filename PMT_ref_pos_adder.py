import numpy as np
import sqlite3 as sql
from typing import List

class ReferencePositionAdder:
    def __init__(self, 
                con_source: sql.Connection,
                source_table: str,
                event_no_subset: List[int],
                tolerance_xy: float = 10,
                tolerance_z: float = 2) -> None:
        self.con_source = con_source
        self.source_table = source_table
        self.event_no_subset = event_no_subset
        self.reference_data = self._load_reference_data('/groups/icecube/cyan/factory/DOMification/dom_ref_pos/unique_string_dom_completed.csv')
        self.tolerance_xy = tolerance_xy
        self.tolerance_z = tolerance_z

    def __call__(self) -> None:
        """
        Callable method to update string and dom_number for a subset of events.
        """
        self._add_columns_if_missing()
        self._update_string_dom_number()

    def _load_reference_data(self, filepath: str) -> np.ndarray:
        """
        Load reference data from a CSV file, skipping the header row.
        """
        return np.loadtxt(filepath, delimiter=',', skiprows=1)

    def _add_columns_if_missing(self) -> None:
        cur_source = self.con_source.cursor()
        cur_source.execute(f"PRAGMA table_info({self.source_table})")
        existing_columns = [col[1] for col in cur_source.fetchall()]
        if 'string' not in existing_columns:
            cur_source.execute(f"ALTER TABLE {self.source_table} ADD COLUMN string INTEGER")
        if 'dom_number' not in existing_columns:
            cur_source.execute(f"ALTER TABLE {self.source_table} ADD COLUMN dom_number INTEGER")
        self.con_source.commit()
    
    def _update_string_dom_number(self) -> None:
        """
        Update the string and dom_number columns based on reference data matching.
        """
        cur_source = self.con_source.cursor()

        # Compute bounding box for filtering reference data
        event_filter = ','.join(map(str, self.event_no_subset))
        query = f"""
            SELECT MIN(dom_x), MAX(dom_x), MIN(dom_y), MAX(dom_y)
            FROM {self.source_table}
            WHERE event_no IN ({event_filter})
        """
        cur_source.execute(query)
        bounds = cur_source.fetchone()

        # Filter relevant reference data
        relevant_data = self._filter_relevant_reference_data(bounds)

        # Fetch rows to update
        rows_to_update = self._fetch_rows_to_update()

        # Batch update rows
        updates = []
        for row in rows_to_update:
            row_id, dom_x, dom_y, dom_z = row
            matches_xy = relevant_data[
                (np.abs(relevant_data[:, 2] - dom_x) <= self.tolerance_xy) &
                (np.abs(relevant_data[:, 3] - dom_y) <= self.tolerance_xy)
            ]
            if len(matches_xy) > 0:
                match_z = matches_xy[np.abs(matches_xy[:, 4] - dom_z) <= self.tolerance_z]
                if len(match_z) > 0:
                    string_val = int(match_z[0, 0])
                    dom_number_val = int(match_z[0, 1])
                    updates.append((string_val, dom_number_val, row_id))

        # Execute batch updates
        cur_source.executemany(
            f"UPDATE {self.source_table} SET string = ?, dom_number = ? WHERE rowid = ?", updates
        )
        self.con_source.commit()
        
    def _filter_relevant_reference_data(self,
                                        bounds: tuple) -> np.ndarray:
        """
        Filter reference data based on bounding box and tolerances.
        """
        min_dom_x, max_dom_x, min_dom_y, max_dom_y = bounds
        return self.reference_data[
            (self.reference_data[:, 2] >= min_dom_x - self.tolerance_xy) &
            (self.reference_data[:, 2] <= max_dom_x + self.tolerance_xy) &
            (self.reference_data[:, 3] >= min_dom_y - self.tolerance_xy) &
            (self.reference_data[:, 3] <= max_dom_y + self.tolerance_xy)
        ]

    def _fetch_rows_to_update(self) -> List[tuple]:
        """
        Fetch rows where string or dom_number is missing for specific events.
        """
        event_filter = ','.join(map(str, self.event_no_subset))
        query = f"""
            SELECT rowid, dom_x, dom_y, dom_z 
            FROM {self.source_table}
            WHERE event_no IN ({event_filter}) 
            AND (string IS NULL OR dom_number IS NULL)
        """
        cur_source = self.con_source.cursor()
        cur_source.execute(query)
        return cur_source.fetchall()