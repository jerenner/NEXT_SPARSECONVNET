#!/usr/bin/env python
"""
This script creates hdf5 files from sensim voxel data that contains:
 - DATASET/BinClassHits - voxelized hits table with labels (binary classification)
 - DATASET/BinInfo      - table that stores info about bins
 - DATASET/EventsInfo   - table that contains EventID and binary classification label
"""
import sys
import os
import tables as tb
import numpy  as np
import pandas as pd

from glob import glob
from invisible_cities.io import dst_io as dio
from invisible_cities.cities.components import index_tables
from invisible_cities.core.configure import configure

def process_sensim_files(file_list, config, start_id=0, energy_min=2.4, energy_max=2.55):
    """
    Process sensim voxel data files and return the voxelized data.
    """
    voxel_file = file_list[0]  # Use the first file to infer voxel dimensions
    with tb.open_file(voxel_file, 'r') as h5in:
        df_voxels_sensim = pd.DataFrame.from_records(h5in.root.Sensim.sns_df[:])

    # Infer the min/max values and spacing for x, y, and z
    unique_x_sipm = np.sort(df_voxels_sensim.x_sipm.unique())
    unique_y_sipm = np.sort(df_voxels_sensim.y_sipm.unique())
    unique_z_slices = np.sort(df_voxels_sensim.z_slice.unique())

    # Compute the min and max values, as well as the uniform spacing for each dimension
    min_x, max_x = -495., 495. #unique_x_sipm.min(), unique_x_sipm.max()
    min_y, max_y = -495., 495. #unique_y_sipm.min(), unique_y_sipm.max()
    min_z, max_z = 0., 1203. #unique_z_slices.min(), unique_z_slices.max()
    print(f"x-slices (min,max) = ({unique_x_sipm.min()},{unique_x_sipm.max()})")
    print(f"y-slices (min,max) = ({unique_y_sipm.min()},{unique_y_sipm.max()})")
    print(f"z-slices (min,max) = ({unique_z_slices.min()},{unique_z_slices.max()})")

    # Infer uniform spacing based on the differences between unique values
    spacing_x = np.diff(unique_x_sipm).mean()*1  # Assuming uniform spacing
    spacing_y = np.diff(unique_y_sipm).mean()*1  # Assuming uniform spacing
    spacing_z = np.diff(unique_z_slices).mean()*4.762 # Assuming uniform spacing 

    # Define bins for voxelization based on inferred min, max, and spacing
    bins_x = np.arange(min_x-spacing_x/2, max_x + 3*spacing_x/2, spacing_x)
    bins_y = np.arange(min_y-spacing_y/2, max_y + 3*spacing_y/2, spacing_y)
    bins_z = np.arange(min_z-spacing_z/2, max_z + 3*spacing_z/2, spacing_z)
    print(f"Found bins_x = {bins_x}, bins_y = {bins_y}, bins_z = {bins_z}")
    sys.stdout.flush()

    bins = (bins_x, bins_y, bins_z)

    eventInfo_list = []
    hits_list = []
    current_dataset_id = start_id

    for i, f in enumerate(file_list):
        print(f"Processing file {i+1}/{len(file_list)}: {f}")

        # Load the sensim voxel data
        with tb.open_file(f, 'r') as h5in:
            df_voxels_sensim = pd.DataFrame.from_records(h5in.root.Sensim.sns_df[:])

        # Filter by energy range if required
        print("-- Filtering on energy...")
        total_energy_per_event = df_voxels_sensim.groupby('event')['energy'].sum().reset_index()
        valid_events = total_energy_per_event[
            (total_energy_per_event['energy'] >= energy_min) & 
            (total_energy_per_event['energy'] <= energy_max)
        ]['event']
        df_voxels_sensim = df_voxels_sensim[df_voxels_sensim['event'].isin(valid_events)]
        print(f"Filtered events: {len(valid_events)} out of total {len(total_energy_per_event)}")

        # Bin the x_sipm, y_sipm, and z_slice into xbin, ybin, zbin
        print(f"-- Creating voxels for min x_sipm = {df_voxels_sensim['x_sipm'].min()}, max x_sipm = {df_voxels_sensim['x_sipm'].max()}...")
        xbin = pd.cut(df_voxels_sensim['x_sipm'], bins_x, labels=np.arange(len(bins_x) - 1)).astype(int)
        ybin = pd.cut(df_voxels_sensim['y_sipm'], bins_y, labels=np.arange(len(bins_y) - 1)).astype(int)
        zbin = pd.cut(df_voxels_sensim['z_slice'], bins_z, labels=np.arange(len(bins_z) - 1)).astype(int)

        df_voxels_sensim = df_voxels_sensim.assign(xbin=xbin, ybin=ybin, zbin=zbin)

        # Group by xbin, ybin, zbin, and event and sum energies
        voxelized_hits = df_voxels_sensim.groupby(['xbin', 'ybin', 'zbin', 'event']).apply(
            lambda df: pd.Series({
                'energy': df['energy'].sum(),
                'binclass': int(df['binclass'].unique()[0])  # Assume consistent binclass
            })
        ).reset_index()

        # Assign dataset_id
        print("-- Assigning unique IDs...")
        eventInfo = voxelized_hits[['event', 'binclass']].drop_duplicates().reset_index(drop=True)
        dct_map = {eventInfo.iloc[i].event: i + current_dataset_id for i in range(len(eventInfo))}
        eventInfo = eventInfo.assign(dataset_id=eventInfo['event'].map(dct_map))

        voxelized_hits = voxelized_hits.assign(dataset_id=voxelized_hits['event'].map(dct_map))
        voxelized_hits = voxelized_hits.drop('event', axis=1)

        # Collect event info and hits
        eventInfo_list.append(eventInfo)
        hits_list.append(voxelized_hits)

        # Update dataset_id for the next file
        current_dataset_id += len(eventInfo)

        sys.stdout.flush()

    # Concatenate all eventInfo and hits
    eventInfo_all = pd.concat(eventInfo_list, ignore_index=True)
    hits_all = pd.concat(hits_list, ignore_index=True)

    # Create binning info
    binsInfo = pd.Series({
        'min_x': min_x, 'max_x': max_x, 'nbins_x': int(len(bins_x) - 1),
        'min_y': min_y, 'max_y': max_y, 'nbins_y': int(len(bins_y) - 1),
        'min_z': min_z, 'max_z': max_z, 'nbins_z': int(len(bins_z) - 1),
        'Rmax': config.Rmax
    }).to_frame().T

    return eventInfo_all, binsInfo, hits_all


if __name__ == "__main__":
    config  = configure(sys.argv).as_namespace
    filesin = glob(os.path.expandvars(config.files_in))
    fout = os.path.expandvars(config.file_out)
    start_id = 0
    if os.path.isfile(fout):
        raise Exception('output file exists, please remove it manually')
    
    # Process sensim files
    eventInfo, binsInfo, hits = process_sensim_files(filesin, config, start_id)

    # Write to output file
    with tb.open_file(fout, 'a') as h5out:
        dio.df_writer(h5out, eventInfo, 'DATASET', 'EventsInfo', columns_to_index=['dataset_id'], str_col_length=64)
        dio.df_writer(h5out, binsInfo , 'DATASET', 'BinsInfo')
        dio.df_writer(h5out, hits     , 'DATASET', 'Voxels', columns_to_index=['dataset_id'])

    # Index tables in the output file
    index_tables(fout)
