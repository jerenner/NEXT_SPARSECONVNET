import os
import tables as tb
import pandas as pd
import numpy as np
from invisible_cities.io import dst_io as dio
from invisible_cities.cities.components import index_tables
import argparse

def combine_files_incremental(input_files, output_file):
    """
    Combine multiple HDF5 files into a single HDF5 file with continuous dataset_id values,
    processing one file at a time to minimize memory usage.
    
    Args:
        input_files (list): List of input HDF5 files to combine.
        output_file (str): Path to the output combined HDF5 file.
    """
    # Prevent overwriting an existing file
    if os.path.isfile(output_file):
        raise Exception(f'Output file {output_file} exists, please remove it manually')
    
    # Initialize cumulative event counter
    total_events = 0
    
    # Flag to track if this is the first file (for BinsInfo and table initialization)
    first_file = True
    
    # Process each file incrementally
    for i, file in enumerate(input_files):
        print(f"Processing file {i+1}/{len(input_files)}: {file}")
        with tb.open_file(file, 'r') as h5in:
            eventsInfo = pd.DataFrame.from_records(h5in.root.DATASET.EventsInfo[:])
            voxels = pd.DataFrame.from_records(h5in.root.DATASET.Voxels[:])
            if first_file:
                binsInfo = pd.DataFrame.from_records(h5in.root.DATASET.BinsInfo[:])
        
        # Adjust dataset_id with the cumulative offset
        eventsInfo['dataset_id'] += total_events
        voxels['dataset_id'] += total_events
        
        # Open output file: 'w' for first file to create, 'a' for subsequent to append
        mode = 'w' if first_file else 'a'
        with tb.open_file(output_file, mode) as h5out:
            if first_file:
                # Write BinsInfo only once
                dio.df_writer(h5out, binsInfo, 'DATASET', 'BinsInfo')
            # Write EventsInfo and Voxels (appends if tables exist)
            dio.df_writer(h5out, eventsInfo, 'DATASET', 'EventsInfo', 
                          columns_to_index=['dataset_id'], str_col_length=64)
            dio.df_writer(h5out, voxels, 'DATASET', 'Voxels', 
                          columns_to_index=['dataset_id'])
        
        # Update the cumulative event count
        total_events += len(eventsInfo)
        first_file = False
    
    # Index tables for efficient querying
    index_tables(output_file)
    print(f"Combined file created: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine multiple HDF5 files into one incrementally.')
    parser.add_argument('input_files', nargs='+', help='List of input HDF5 files to combine')
    parser.add_argument('output_file', help='Output combined HDF5 file')
    args = parser.parse_args()
    
    combine_files_incremental(args.input_files, args.output_file)
