import numpy as np
import sqlite3 as sql
from typing import List
import pandas as pd
import logging

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
        self._create_indexes()
        
    def _create_indexes(self) -> None:
        cur_source = self.con_source.cursor()
        indexes = [
            ("idx_event_no", "event_no"),
            ("idx_dom_position", "dom_x, dom_y, dom_z"),
            ("idx_string_dom_number", "string, dom_number")
        ]
        for idx_name, columns in indexes:
            try:
                logging.info(f"Creating index {idx_name} on {columns}.")
                cur_source.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {self.source_table} ({columns})")
            except sql.Error as e:
                logging.error(f"Failed to create index {idx_name}: {e}")
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
        rows_to_update_df = self._fetch_rows_to_update()
        rows_to_update = rows_to_update_df.to_records(index=False).tolist()

        # Extract columns for rows to update
        row_ids, dom_xs, dom_ys, dom_zs = np.array(rows_to_update).T

        # Vectorised matching for XY tolerances
        xy_matches = (
            (np.abs(relevant_data[:, 2][:, None] - dom_xs) <= self.tolerance_xy) &
            (np.abs(relevant_data[:, 3][:, None] - dom_ys) <= self.tolerance_xy)
        )

        # Filter relevant_data for matches
        matching_relevant_data = relevant_data[np.any(xy_matches, axis=1)]

        # Vectorised matching for Z tolerance
        z_matches = np.abs(matching_relevant_data[:, 4][:, None] - dom_zs) <= self.tolerance_z

        # Get matching rows
        matched_rows = np.where(z_matches)

        # Extract string and dom_number for matches
        matched_strings = matching_relevant_data[matched_rows[0], 0]
        matched_dom_numbers = matching_relevant_data[matched_rows[0], 1]

        # Combine row_id with matches for batch update
        updates = list(zip(matched_strings, matched_dom_numbers, row_ids[matched_rows[1]]))

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
        tol_xy = self.tolerance_xy
        return self.reference_data[
            np.logical_and.reduce((
                self.reference_data[:, 2] >= min_dom_x - tol_xy,
                self.reference_data[:, 2] <= max_dom_x + tol_xy,
                self.reference_data[:, 3] >= min_dom_y - tol_xy,
                self.reference_data[:, 3] <= max_dom_y + tol_xy
            ))
        ]

    def _fetch_rows_to_update(self) -> pd.DataFrame:
        event_filter = ','.join(map(str, self.event_no_subset))
        query = f"""
            SELECT rowid, dom_x, dom_y, dom_z 
            FROM {self.source_table}
            WHERE event_no IN ({event_filter}) 
            AND (string IS NULL OR dom_number IS NULL)
        """
        # Fetch rows directly into a Pandas DataFrame
        rows_to_update_df = pd.read_sql_query(query, self.con_source)

        return rows_to_update_df