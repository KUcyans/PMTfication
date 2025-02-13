import pyarrow as pa
import pyarrow.parquet as pq
import os
import abc
import pyarrow.compute as pc
import logging
from functools import wraps
import json
import time

class EventFilter(abc.ABC):
    def __init__(self, source_dir: str, output_dir: str, subdir_no: int, part_no: int, **kwargs):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.subdir_no = subdir_no
        self.part_no = part_no
        self.valid_event_nos = set()
        self.extra_params = kwargs
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = logging.getLogger(self.__class__.__name__) 
        self.logger.info(f"Initialized {self.__class__.__name__}")

    def __call__(self):
        self.logger.info(f"Starting filtering process for {self.subdir_no}/{self.part_no}")
        self._filter_truth()
        self._filter_shards()
        self.logger.info("Filtering process completed.")

    @abc.abstractmethod
    def _filter_truth(self):
        # override this method in subclasses
        pass
    
    def _recalculate_offset(self, truth_table: pa.Table) -> pa.Table:
        if "N_doms" not in truth_table.column_names:
            self.logger.error("Column 'N_doms' is missing from truth table. Cannot recalculate 'offset'.")
            return truth_table
        
        truth_data = {col: truth_table.column(col).to_pylist() for col in truth_table.column_names}
        truth_data['offset'] = pc.cumulative_sum(pa.array(truth_data['N_doms']))
        
        self.logger.info("Recalculated 'offset' column based on filtered 'N_doms' values.")
        return pa.Table.from_pydict(truth_data)

    
    def _filter_shards(self):
        """Filters PMTfied files using the extracted valid event numbers."""
        source_pmtfied_dir = os.path.join(self.source_dir, str(self.part_no))
        dest_pmtfied_dir = os.path.join(self.output_dir, str(self.part_no))
        os.makedirs(dest_pmtfied_dir, exist_ok=True)

        for file in os.listdir(source_pmtfied_dir):
            source_pmtfied_file = os.path.join(source_pmtfied_dir, file)
            dest_pmtfied_file = os.path.join(dest_pmtfied_dir, file)

            pmt_table = pq.read_table(source_pmtfied_file)
            pmt_event_nos = pmt_table.column("event_no").to_pylist()
            valid_indices = [i for i, eno in enumerate(pmt_event_nos) if eno in self.valid_event_nos]

            if valid_indices:
                filtered_pmt_table = pmt_table.take(valid_indices)
                pq.write_table(filtered_pmt_table, dest_pmtfied_file)
                self.logger.info(f"Filtered PMTfied file saved to: {dest_pmtfied_file}")
            else:
                self.logger.warning(f"Skipping {file}: No valid events found in PMTfied file.")

    def _generate_receipt(self, source_truth_file: str, output_truth_file: str):
        if not os.path.isfile(source_truth_file):
            self.logger.error(f"Truth file not found: {source_truth_file}")
            return
        if not os.path.isfile(output_truth_file):
            self.logger.error(f"Output file not found: {output_truth_file}")
            return
        
        initial_event_count = pq.read_table(source_truth_file).num_rows
        reduced_event_count = pq.read_table(output_truth_file).num_rows
        
        receipt_file = os.path.join(self.output_dir, f"[Receipt]{self.subdir_no}_{self.part_no}.json")
        
        survival_ratio = reduced_event_count/initial_event_count
        receipt_data = {
            "subdir_no": self.subdir_no,
            "part_no": self.part_no,
            "initial_event_count": initial_event_count,
            "selected_event_count": reduced_event_count,
            "surviving_percentage": round(100 * survival_ratio, 4),
            "reduced_percentage": round(100 * (1 - survival_ratio), 4),
            "start_time": None,
            "end_time": None,
            "execution_duration": None,
        }

        with open(receipt_file, "w") as f:
            json.dump(receipt_data, f, indent=4)
        self.logger.info(f"Receipt file saved to: {receipt_file}")
        
    def update_receipt_time(self, start_time: float, end_time: float, duration: float):
        """Updates the receipt file with execution start time, end time, and duration."""
        receipt_file = os.path.join(self.output_dir, f"[Receipt]{self.subdir_no}_{self.part_no}.json")

        if not os.path.exists(receipt_file):
            self.logger.error(f"Receipt file not found: {receipt_file}. Cannot update timing info.")
            return
        
        # Read existing JSON
        with open(receipt_file, "r") as f:
            receipt_data = json.load(f)

        # Update timestamp and execution time
        receipt_data["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        receipt_data["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
        receipt_data["execution_duration"] = round(duration, 4)

        # Save updated receipt
        with open(receipt_file, "w") as f:
            json.dump(receipt_data, f, indent=4)

        self.logger.info(f"Updated receipt with execution time: {receipt_file}")

def override(method):
    """Decorator to indicate a method overrides a superclass method."""
    @wraps(method)
    def wrapper(*args, **kwargs):
        return method(*args, **kwargs)
    return wrapper
