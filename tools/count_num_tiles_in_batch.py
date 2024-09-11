
"""
    Script for counting number of tiles (of all images in a batch dataset).
    
    can be used to sample images from any npy dataset.
    
"""


from multiprocessing import Pool
import numpy as np
import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.lib.image_sampling_utils import sample_images_all_markers_all_lines, sample_images_all_markers

BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
INPUT_DIR = os.path.join(BASE_DIR,'input','images','processed','spd2','SpinningDisk')


def _count_tiles(img_path):
        
        # load the npy (16 tiles, 2 channels, 100x100)
        x = np.load(img_path)
        
        return x.shape[0]
    
def calc_num_tiles_neurons_batch(batch_num, sample_size_per_markers=200, num_markers=26, exclude_DAPI=True):
    
    print(f"\ncalc_num_tiles_neurons_batch {batch_num}")
    
        
    # Global paths
    BATCH_TO_RUN = 'batch'+str(batch_num)
    INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN)

    images = sample_images_all_markers_all_lines(INPUT_DIR_BATCH, 
                                                 sample_size_per_markers, 
                                                 num_markers, 
                                                 all_conds=True,
                                                 exclude_DAPI=exclude_DAPI
                                                 )
                                                 #markers_to_include=['DAPI']
    
    n_images  = len(images)
    print("\n\nTotal of", n_images, "images were sampled.")
    
    total_num_tiles = 0
    files = [img_path for img_path in images]
    with Pool() as mp_pool:    
        
        for count in mp_pool.map(_count_tiles, files):
            total_num_tiles+=count
        
        mp_pool.close()
        mp_pool.join()
    
    print(f"\n\nTotal number of tiles in {BATCH_TO_RUN}: {total_num_tiles}")
    return total_num_tiles

if __name__ == '__main__':
    print("\n\n\nStart..")
    
    
    calc_num_tiles_neurons_batch(batch_num='6', 
                                 sample_size_per_markers=200, 
                                 num_markers=26,
                                 exclude_DAPI=True)
    
    calc_num_tiles_neurons_batch(batch_num='6_16bit_no_downsample', 
                                 sample_size_per_markers=200, 
                                 num_markers=26,
                                 exclude_DAPI=True)
    
    
    calc_num_tiles_neurons_batch(batch_num='6_add_brenner_cellpose_wo_frame', 
                                 sample_size_per_markers=200, 
                                 num_markers=26,
                                 exclude_DAPI=True)
    
    calc_num_tiles_neurons_batch(batch_num='6_original_with_brenner', 
                                 sample_size_per_markers=200, 
                                 num_markers=26,
                                 exclude_DAPI=True)
    
    
    

    
    
    print("\n\n\n\nDone!")

