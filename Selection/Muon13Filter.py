import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
import os

from EventFilter import EventFilter, override

class Muon13Filter(EventFilter):
    def __init__(self, 
                 source_dir: str, 
                 output_dir: str, 
                 subdir_no: int, 
                 part_no: int):
        super().__init__(source_dir=source_dir, 
                         output_dir=output_dir, 
                         subdir_no=subdir_no, 
                         part_no=part_no)
        
    def __call__(self):
        self.logger.info(f"Starting filtering process for {self.subdir_no}/{self.part_no}")
        self._filter_truth()
        self._filter_shards()
        self.logger.info("Filtering process completed.")
    
    @override    
    def _filter_truth(self):
        """Filters truth file based on MuonFilter_13 and writes it back."""
        source_truth_file = os.path.join(self.source_dir, str(self.subdir_no), f"truth_{self.part_no}.parquet")
        if not os.path.isfile(source_truth_file):
            self.logger.error(f"Truth file not found: {source_truth_file}")
            return

        truth_table = pq.read_table(source_truth_file)
        filtered_truth_table = self._apply_event_filter(truth_table)

        if filtered_truth_table is None:
            return

        output_truth_file = os.path.join(self.output_dir, str(self.subdir_no), f"truth_{self.part_no}.parquet")
        pq.write_table(filtered_truth_table, output_truth_file)
        self.logger.info(f"Filtered truth file saved to: {output_truth_file}")
        
        self._generate_receipt(source_truth_file, output_truth_file)

    def _apply_event_filter(self, truth_table: pa.Table) -> pa.Table:
        """Filters the truth table to include only events where MuonFilter_13 is True."""
        required_columns = {"MuonFilter_13", "N_doms", "event_no"}
        missing_columns = required_columns - set(truth_table.column_names)
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}. Cannot proceed with filtering.")
            return None

        # Convert MuonFilter_13 to boolean values if needed
        muon_filter = truth_table.column("MuonFilter_13")
        valid_indices = pc.equal(muon_filter, True).to_numpy().nonzero()[0]

        if len(valid_indices) == 0:
            self.logger.warning(f"No valid MuonFilter_13 events found in {self.subdir_no}/{self.part_no}. Skipping filtering.")
            return None

        # apply filter
        filtered_table = truth_table.take(valid_indices)
        self.valid_event_nos = set(filtered_table.column("event_no").to_pylist())
        filtered_table = self._recalculate_offset(filtered_table)

        return filtered_table