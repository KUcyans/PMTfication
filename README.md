# DOM-wise Summarisation for IceCube Monte Carlo Data
> python files a user runs: [PMTfy_by_part.py(recommended)](https://github.com/KUcyans/PMTfication/blob/main/PMTfy_by_part.py) of `PMTfy_by_subdir.py`.

## Assumptions and the Data Storage Structure:
1. By Jan. 2025, only summarisation **from sqlite .db to .parquet** format is supported
2. Source Data(.db):
   * Data are stored across several directories.
   * Each directory(e.g.22012) has several multiple .db files each has a conceptual name of 'part' in this project.
3. Target Data(.parquet)
   * The structure in which the target data are stored is similar to the source data.
   * There will be the directory having the same name as the source directory(e.g.22012)
   * In each directory, there will be several subdirectories in each directory. The subdirectories correspond to the 'parts'(the .db files in the source directory)
   * In each directory, there will also be truth parquet files and they also correspond to the 'parts'
   * In each subdirectory, there will be DOM-wise summarised files. Each of them are called 'shard' in the code.
![image](https://github.com/user-attachments/assets/cebfef9b-aa21-424d-9a52-3bb9c8df39ff)

## Logical structure of the code
  * Core summarisation functions are in [PMT_summariser](https://github.com/KUcyans/PMTfication/blob/main/PMT_summariser.py): it is the class that governs the calculation of the data
    * uses PMT_ref_pos_adder
  * [PMT_ref_pos_adder](https://github.com/KUcyans/PMTfication/blob/main/PMT_ref_pos_adder.py) adds `string` and `dom_number` columns to the data if they are missing.
    * `string` and `dom_number` columns are needed to identify distinctive DOM locations based on `dom_x`, `dom_y`, and `dom_z` with respect to the [sample(reference) DOM positional data](https://github.com/KUcyans/PMTfication/blob/main/dom_ref_pos/unique_string_dom_completed.csv)
  * [PMT_truth_maker](https://github.com/KUcyans/PMTfication/blob/main/PMT_truth_maker.py) generates the truth parquet files for a part
  * [PMTfier](https://github.com/KUcyans/PMTfication/blob/main/PMTfier.py) wraps the core logic files.
    * Consists of three hierarchical layers: shard, part, subdir. 
    * PMT_summariser is in the lowest level, shard
    * PMT_truth_maker is on one more level above, part: (this is why truth files are stored part-wisely)
    * it generates a new event_no based on the location of the data so that every even_no is distinctive from one another. See `_add_enhance_event_no` function.
![image](https://github.com/user-attachments/assets/bdcfb4d1-30b0-486e-a1f5-995e1802b1f3)

## Execution
  * Can be submitted to Slurm if available. See [this shell script](https://github.com/KUcyans/PMTfication/blob/main/PMTfy_by_part.sh)
  * Arguments for [PMTfy_by_part.py](https://github.com/KUcyans/PMTfication/blob/main/PMTfy_by_part.py) are these
    * `Snowstorm_or_Corsika`: can either be Snowstorm or Corsika
    * `subdirectory_in_number`: 22010, 22011, 22012, ... this is the name of the directory where the source data are stored.
    * `part_number`: the number of the file in the directory that will be processed
    * `N_events_per_shard`: the amount of the events in one DOM-wisely summarised file has to be specified.
      * The size of the source data for an event will vary depending on the overall energy range of the data.
      * In general, higher energy data will contain more number of DOMs and more number of pulses at each DOM.
    * the name of the source table from which the data will be loaded is set to be 'SRTInIcePulses'

## Issues
1. For data having too many events in one file, [PMTfy_by_part.py](https://github.com/KUcyans/PMTfication/blob/main/PMTfy_by_part.py) may fail to process the data.

## Tasks
> * DO NOT use pandas for final implementation.
> * Keeping track of the hierarchical structure of event-DOM-pulse in [PMT_summariser](https://github.com/KUcyans/PMTfication/blob/main/PMT_summariser.py) is often frustratingly painful for me. Be patient unless you have an ingenious idea to improve.
1. [PMT_truth_maker](https://github.com/KUcyans/PMTfication/blob/main/PMT_truth_maker.py)
  1. cannot handle data files whose truth table do not have exactly the same column names as specified in the schemas.
    * schemas are meant to make truth-making more efficient but it is not suitable to cope with different source data.
    * So one might have to consider avoid using schemas or implement more flexible schema building logic
2. [PMT_ref_pos_adder](https://github.com/KUcyans/PMTfication/blob/main/PMT_ref_pos_adder.py)
  1. currently uses .csv file but it could be replaced with a GeoCalib file. (Leave the csv file as default option)  
3. [PMT_summariser](https://github.com/KUcyans/PMTfication/blob/main/PMT_summariser.py)
  1. [Feasibility check needed before instigation]change the list structure to np.array if possible.
  2. Possible parallelisation or threading of the for loops
4. [PMTfier](https://github.com/KUcyans/PMTfication/blob/main/PMTfier.py)
  1. The current implementation is highly specific to the data stored structure that KU HEP has. There should be more general
  2. Replacing the functions with classes might facilitate the maintenance
