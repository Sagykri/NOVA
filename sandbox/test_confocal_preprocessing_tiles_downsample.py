from datetime import datetime
from glob import glob 
import logging
import random
import os
import sys


import numpy as np
from skimage import io, transform
from skimage.measure import block_reduce
import matplotlib.pyplot as plt

import sys
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # to be able import parent directory
from src.common.lib import preprocessing_utils

# Global paths
BATCH_TO_RUN = 'batch3' 

BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
INPUT_DIR = os.path.join(BASE_DIR,'input','images','raw','220814_neurons', 'TDP43', 'unstressed')


def save_tile(img, save_path):
    
    # Increase contrast with min-max scaling
    #img = (img - img.min(axis=0)) / (img.max(axis=0) - img.min(axis=0)) # TODO: talk to Sagy about this, with this you see horizental lines
    
    # save image to "save_path" (full path + file name)
    plt.imsave(save_path, img, cmap = 'rainbow')
    return None


def main(_tiles_size = 256, sample_size_per_markers=1, num_markers=10):
    
    c_0 = 'DAPI'
    c_1 = 'G3BP1'
    c_2 = 'TIA1'
    c_3 = 'KIF5A'
    channels = [c_0, c_1, c_2, c_3]
    
    file_path = os.path.join(INPUT_DIR, "220811_iNDI_TDP43_unstressed-"+c_0+"_"+c_1+"_"+c_2+"_"+c_3+"-01.tif")
    img = io.imread(file_path)
    print("raw image shape:", img.shape)
    
    n_channels = img.shape[-1]
    img_processed = None
    for c in range(n_channels):
        img_current_channel = np.array(img[...,c], dtype=np.float64)
        
        # Downsampling
        img_current_channel = preprocessing_utils.downsample(show=False, 
                                             img_current_channel=img_current_channel,
                                             block_size=2) 
        
        #print("img_current_channel shape", img_current_channel.shape)
    

        # Normalize
        preprocessing_utils.normalize(show=False, img_current_channel=img_current_channel)

        if img_processed is None:
         img_processed = img_current_channel
        else:
            img_processed = np.dstack((img_processed, img_current_channel))
        
    print("Image (post image processing) shape", img_processed.shape, flush=True)

    # Crop tiles
    image_processed_tiles = preprocessing_utils.crop_to_tiles(tile_w=300, tile_h=300, img_processed=img_processed)
    print("image_processed_tiles shape:", image_processed_tiles.shape, len(image_processed_tiles), image_processed_tiles[0].shape)
    
    size = 100
    if image_processed_tiles.shape[1] != size:
        # Reshape to sizexsize (100x100)
        print(f"Reshape tiles to {size}x{size} (block_size={image_processed_tiles.shape[1] // size})")
        
        image_processed_tiles_reshaped = preprocessing_utils.rescale(n_channels, image_processed_tiles, size)
    else:
        image_processed_tiles_reshaped = image_processed_tiles

    # Save processed tiles to png files
    
    for c in range(n_channels):
        img_current_channel_tiles = np.array(image_processed_tiles_reshaped[...,c], dtype=np.float64)
        marker_name = channels[c]
        print(marker_name, c)
        for i, tile in enumerate(img_current_channel_tiles):
            # save the tile to a png file
            save_path = os.path.join(BASE_DIR,'sandbox','test_confocal_tiles_for_Lena', marker_name+ "_tile_"+str(i)+".png")
            save_tile(tile, save_path)
            #print("save", save_path)
        
    return None        

if __name__ == '__main__':
    print("\n\n\n\n\nStart..")
    main(_tiles_size = 256, sample_size_per_markers=1, num_markers=1)
    print("\n\n\n\nDone!")
    
    
