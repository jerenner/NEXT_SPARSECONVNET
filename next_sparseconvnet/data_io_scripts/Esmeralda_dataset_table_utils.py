"""
This script creates hdf5 files that contains:
 - DATASET/BinClassHits - voxelized hits table with labels (binary classification)
 - DATASET/SegClassHits - voxelized hits table with labels (segmentation)
 - DATASET/BinInfo      - table that stores info about bins
 - DATASET/EventsInfo   - table that contains EventID and binary classification label
"""
import sys
import os
import tables as tb
import numpy  as np
import pandas as pd

from glob import glob
from invisible_cities.io import mcinfo_io as mio
from invisible_cities.io import dst_io    as dio

from invisible_cities.core.configure import configure

from . import dataset_labeling_utils as utils


def get_CHITtables(esmeralda_output, config, start_id = 0, energy_min = 2.40, energy_max = 2.55, mc_data_factor = 1.0):
    """
    This function processes CHITS from Esmeralda output and voxelizes them for the sparse CNN.
    """

    # Define bins for voxelization
    min_x, max_x = config.xlim
    min_y, max_y = config.ylim
    min_z, max_z = config.zlim
    bins_x = np.linspace(min_x, max_x, config.nbins_x)
    bins_y = np.linspace(min_y, max_y, config.nbins_y)
    bins_z = np.linspace(min_z, max_z, config.nbins_z)
    bins = (bins_x, bins_y, bins_z)

    # Read the CHITS from the Esmeralda file
    chits_events_df = None
    with tb.open_file(esmeralda_output, 'r') as hdf:
        #chits_events_df = pd.DataFrame.from_records(hdf.root['CHITS']['highTh'][:])
        chits_events_df = pd.DataFrame.from_records(hdf.root['RECO']['Events'][:])  # for Sophronia hits

    # Select only relevant columns (e.g., X, Y, Z, energy)
    chits_events_df = chits_events_df[['event', 'X', 'Y', 'Z', 'Ec']].rename(columns={'event': 'event_id', 'X': 'x', 'Y': 'y', 'Z': 'z', 'Ec': 'energy'})

    # Add classification labels.
    pathname, basename = os.path.split(esmeralda_output)
    chits_events_df = utils.add_clf_labels_filename(chits_events_df, basename)

    # -----------------------------------------------------------------------------
    # Group by event_id and calculate the total energy per event
    total_energy_per_event = chits_events_df.groupby('event_id')['energy'].sum().reset_index()

    # Filter events with total energy in the specified range
    valid_events = total_energy_per_event[
        (total_energy_per_event['energy']*mc_data_factor >= energy_min) &
        (total_energy_per_event['energy']*mc_data_factor <= energy_max)
    ]['event_id']
    print(f"Filtered events: {len(valid_events)} out of total {len(total_energy_per_event)}")

    # Filter the chits DataFrame to only include events in the energy range
    chits_events_df = chits_events_df[chits_events_df['event_id'].isin(valid_events)]
    # -----------------------------------------------------------------------------

    # Bin the hits into voxels
    chits_events_df = utils.get_bin_indices(chits_events_df, bins, Rmax=config.Rmax)
    chits_events_df = chits_events_df.sort_values('event_id')

    # Prepare event information and assign unique dataset IDs
    eventInfo = chits_events_df[['event_id', 'binclass']].drop_duplicates().reset_index(drop=True)
    dct_map = {eventInfo.iloc[i].event_id: i + start_id for i in range(len(eventInfo))}
    eventInfo = eventInfo.assign(dataset_id=eventInfo.event_id.map(dct_map))

    # Add dataset_id to hits and drop event_id
    chits_events_df = chits_events_df.assign(dataset_id=chits_events_df.event_id.map(dct_map))
    chits_events_df = chits_events_df.drop('event_id', axis=1)

    # Create binning info
    binsInfo = pd.Series({'min_x': min_x, 'max_x': max_x, 'nbins_x': config.nbins_x,
                          'min_y': min_y, 'max_y': max_y, 'nbins_y': config.nbins_y,
                          'min_z': min_z, 'max_z': max_z, 'nbins_z': config.nbins_z,
                          'Rmax': config.Rmax}).to_frame().T

    return eventInfo, binsInfo, chits_events_df
