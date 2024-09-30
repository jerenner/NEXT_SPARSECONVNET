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


def get_MCtables(filename, config, start_id=0):
    pathname, basename = os.path.split(filename)
    min_x, max_x = config.xlim
    min_y, max_y = config.ylim
    min_z, max_z = config.zlim
    bins_x = np.linspace(min_x, max_x, config.nbins_x)
    bins_y = np.linspace(min_y, max_y, config.nbins_y)
    bins_z = np.linspace(min_z, max_z, config.nbins_z)
    bins = (bins_x, bins_y, bins_z)

    hits      = mio.load_mchits_df(filename)
    #select ACTIVE hits
    hits = hits[hits.label=='ACTIVE']
    particles = mio.load_mcparticles_df(filename)

    if config.classification and config.segmentation:
        hits = utils.add_clf_seg_labels(hits, particles, delta_t=config.blob_delta_t, delta_e=config.blob_delta_e)
    elif config.classification:
        #hits = utils.add_clf_labels(hits, particles)
        hits = utils.add_clf_labels_filename(hits, basename)
    elif config.segmentation:
        hits = utils.add_seg_labels(hits, particles, delta_t=config.blob_delta_t, delta_e=config.blob_delta_e)
    else:
        hits = hits.reset_index()[['event_id', 'x', 'y', 'z', 'energy']]
    hits = utils.get_bin_indices(hits, bins, Rmax=config.Rmax)
    hits = hits.sort_values('event_id')
    eventInfo = hits[['event_id', 'binclass']].drop_duplicates().reset_index(drop=True)
    #create new unique identifier
    dct_map = {eventInfo.iloc[i].event_id : i+start_id for i in range(len(eventInfo))}
    #add dataset_id, pathname and basename to eventInfo
    eventInfo = eventInfo.assign(pathname = pathname, basename = basename, dataset_id = eventInfo.event_id.map(dct_map))
    #add dataset_id to hits and drop event_id
    hits = hits.assign(dataset_id = hits.event_id.map(dct_map))
    hits = hits.drop('event_id', axis=1)

    binsInfo = pd.Series({'min_x'   : min_x ,
                          'max_x'   : max_x ,
                          'nbins_x' : config.nbins_x,
                          'min_y'   : min_y ,
                          'max_y'   : max_y ,
                          'nbins_y' : config.nbins_y,
                          'min_z'   : min_z ,
                          'max_z'   : max_z ,
                          'nbins_z' : config.nbins_z,
                          'Rmax'    : config.Rmax
                          }).to_frame().T

    return eventInfo, binsInfo, hits
