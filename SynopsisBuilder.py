import os
import sqlite3 as sql
from tqdm import tqdm
from typing import List
import argparse
import time
import logging

class SynopsisBuilder:
    def __init__(self, input_dir: str, output_dir: str, source_table: str):
        """
        Initializes the SynopsisBuilder with input and output directories and the source table name.

        Args:
            input_dir (str): Path to the input directory containing .db files.
            output_dir (str): Path to the output directory for saving synopsis .db files.
            source_table (str): The table name in the source database to process.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.source_table = source_table

    def get_numeric_subdirs(self, directory_path: str) -> List[int]:
        """
        Converts the names of subdirectories in the specified directory to integers, 
        filtering out any non-numeric entries.
        
        Args:
            directory_path (str): Path to the main directory containing subdirectories.
            
        Returns:
            List[int]: A list of numeric subdirectory names as integers.
        """
        subdirs = os.listdir(directory_path)
        numeric_subdirs = [int(subdir) for subdir in subdirs if subdir.isdigit()]
        return sorted(numeric_subdirs)

    def shift_event_no(self, event_no: int, subdirectory: int, shift_bits: int = 24) -> int:
        """
        Generate a unique event number by combining event_no and subdirectory.
        
        Args:
            event_no (int): The original event number.
            subdirectory (int): The identifier for the subdirectory.
            shift_bits (int): Number of bits to shift the subdirectory by (default is 24).
            
        Returns:
            int: A unique integer identifier for the event.
        """
        return (subdirectory << shift_bits) | event_no

    def get_event_details(self, cursor, event_no: int) -> tuple:
        """
        Retrieves start and end offsets for a given event_no from the specified table.
        
        Args:
            cursor: Database cursor to execute queries.
            event_no (int): The event number to search for.
            
        Returns:
            tuple: (start_offset, end_offset) for the given event_no.
        """
        cursor.execute(f"SELECT MIN(rowid), MAX(rowid) FROM {self.source_table} WHERE event_no = ?", (event_no,))
        start_offset, end_offset = cursor.fetchone()
        return start_offset, end_offset

    def build_synopsis_file(self, source_db_path: str, synopsis_db_path: str, subdirectory: int):
        """
        Reads data from a source .db file, generates a unique event_no and synopsis, 
        and saves it to a new synopsis .db file.

        Args:
            source_db_path (str): Path to the source .db file.
            synopsis_db_path (str): Path to the output synopsis .db file.
            subdirectory (int): The identifier for the subdirectory.
        """
        # Connect to source database
        conn = sql.connect(source_db_path)
        cursor = conn.cursor()
        
        # Get file name from the path
        file_name = os.path.basename(source_db_path)
        
        # Retrieve unique event numbers from the source database
        cursor.execute(f"SELECT DISTINCT event_no FROM {self.source_table}")
        event_nos = [row[0] for row in cursor.fetchall()]

        # Connect to the synopsis database and create the table structure
        synopsis_conn = sql.connect(synopsis_db_path)
        synopsis_cursor = synopsis_conn.cursor()
        
        synopsis_cursor.execute("""
            CREATE TABLE IF NOT EXISTS synopsis (
                event_no INTEGER PRIMARY KEY,
                file_name TEXT,
                start_offset INTEGER,
                end_offset INTEGER
            )
        """)

        # Process each unique event_no and insert into the synopsis database with a progress bar
        for event_no in tqdm(event_nos, desc=f"Processing {file_name}", unit="event"):
            # Generate unique event_no (shifted version)
            event_no = self.shift_event_no(event_no, subdirectory)
            
            # Get event details: start_offset and end_offset
            start_offset, end_offset = self.get_event_details(cursor, event_no)
            
            # Insert data into synopsis database
            synopsis_cursor.execute("""
                INSERT INTO synopsis (event_no, file_name, start_offset, end_offset)
                VALUES (?, ?, ?, ?)
            """, (event_no, file_name, start_offset, end_offset))

        # Commit and close connections
        synopsis_conn.commit()
        synopsis_conn.close()
        conn.close()
        print(f"Synopsis file created at {synopsis_db_path}")

    def process_all_files_in_dir(self):
        """
        Iterates over all .db files in the input directory and processes each one to create a synopsis file.
        Skips files that already have completed synopsis files.
        """
        print(f"Processing files in {self.input_dir}")
        
        # List all .db files directly in the input directory
        db_files = [f for f in os.listdir(self.input_dir) if f.endswith('.db')]
        if not db_files:
            print(f"No .db files found in {self.input_dir}. Exiting.")
            return

        print(f"Found {len(db_files)} .db files in {self.input_dir}: {db_files}")

        for db_file in db_files:
            source_db_path = os.path.join(self.input_dir, db_file)
            synopsis_db_path = os.path.join(self.output_dir, db_file.replace('.db', '_synopsis.db'))
            
            # Check if the synopsis file already exists and is complete
            if os.path.exists(synopsis_db_path):
                print(f"Checking synopsis for {db_file}: Exists = {os.path.exists(synopsis_db_path)}")
                try:
                    # Check if the existing synopsis has all the required events
                    synopsis_conn = sql.connect(synopsis_db_path)
                    synopsis_cursor = synopsis_conn.cursor()
                    synopsis_cursor.execute("SELECT COUNT(*) FROM synopsis")
                    completed_events = synopsis_cursor.fetchone()[0]
                    synopsis_conn.close()

                    # Check the total number of unique event_no in the source file
                    conn = sql.connect(source_db_path)
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT COUNT(DISTINCT event_no) FROM {self.source_table}")
                    total_events = cursor.fetchone()[0]
                    conn.close()
                    
                    print(f"File {db_file}: Completed events = {completed_events}, Total events = {total_events}")
                    
                    # If the synopsis file already has all events, skip processing
                    if completed_events >= total_events:
                        print(f"Skipping {db_file}: synopsis already complete.")
                        continue
                except Exception as e:
                    print(f"Error checking {synopsis_db_path}: {e}")
            
            # Measure time for each synopsis file creation
            print(f"Starting to process {db_file}")
            start_time = time.time()
            try:
                self.build_synopsis_file(source_db_path, synopsis_db_path, subdirectory=0)  # Set subdirectory to 0 or ignore
            except Exception as e:
                print(f"Failed to create synopsis for {db_file}: {e}")
                continue
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Print the time taken for each file
            print(f"Time taken to create {db_file}_synopsis: {elapsed_time:.2f} seconds")

            
if __name__ == "__main__":
    # Setup argument parser
    print("Script started...")
    logging.info("Script started...")
    parser = argparse.ArgumentParser(description="Generate synopsis files from input directory of .db files.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing .db files.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory where synopsis files will be stored.")
    
    # Hard-coded source table name
    source_table_name = "SRTInIcePulses"

    # Parse arguments
    args = parser.parse_args()
    
    # Print parsed arguments for debugging
    print(f"Arguments received: input_dir={args.input_dir}, output_dir={args.output_dir}")
    
    # Instantiate the SynopsisBuilder class
    builder = SynopsisBuilder(args.input_dir, args.output_dir, source_table_name)
    
    # Start processing and measure time for each synopsis creation
    builder.process_all_files_in_dir()
    
    print("Script finished successfully.")
    logging.info("Script finished successfully.")

