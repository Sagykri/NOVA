# TODO: MOVE TO A DIFFERENT FILE/FOLDER + UTILIZE CONFIGURATION (SAGY WROTE THIS TODO)

"""
    Script for calculating variance of tiles of all images in a batch dataset.
    
    In general, "images = sample_images_all_markers_all_lines(sample_size_per_markers=XX)"
    and "sample_images_all_markers(cell_line_path, sample_size_per_markers=sample_size_per_markers, num_markers=XX)"
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

    
def _calc_variance(img_path):

    # load the npy (16 tiles, 2 channels, 100x100)
    x = np.load(img_path)
    # take only protein channel
    x = x[:,:,:,0]

    tiles_var = np.var(x, axis=(1,2)).mean()
    #return np.sum(x, axis=(1,2)), np.sum(np.power(x ,2), axis=(1,2)), x.shape[0]
    return tiles_var 

def _multiproc_calc_variance(images_paths):
    
    images = images_paths
    n_images  = len(images)
    print("\n\nTotal of", n_images, "images were sampled.")
    
    vars = []
    with Pool() as mp_pool:    
        for mean_var in mp_pool.map(_calc_variance, ([img_path for img_path in images])):
            vars.append(mean_var) 
        mp_pool.close()
        mp_pool.join()
    
    print(f"Variance: {np.mean(vars)}")
    return np.mean(vars)

def calc_variance_neurons_batch(batch_num):
    # Global paths
    BATCH_TO_RUN = 'batch'+str(batch_num)
    INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN)

    images = sample_images_all_markers_all_lines(INPUT_DIR_BATCH, sample_size_per_markers=200, num_markers=26)
    
    variance = _multiproc_calc_variance(images_paths=images)
    
    return variance

def calc_variance_opencell():
    """
    This script was last used (July 10) for calculating variance of tiles of all images in the OpenCell dataset.
    """
    # Global paths
    BATCH_TO_RUN = 'OpenCell' 
    INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN)

    images = sample_images_all_markers(cell_line_path=INPUT_DIR_BATCH, sample_size_per_markers=400, num_markers=1311, depth=3)
    variance = _multiproc_calc_variance(images_paths=images)
    
    return variance


if __name__ == '__main__':
    print("\n\n\nStart..")
    
    calc_variance_neurons_batch(batch_num=7)
    
    #calc_variance_opencell()
    
    print("\n\n\n\nDone!")

# Batch 8
# Total of 40894 images were sampled.
# Variance: 0.0013192599553113191
# Variance:  0.0014811185558576319
# Variance: 0.001349994498773231
# Variance: 0.001465275497971026

# Batch 6
# Total of 38963 images were sampled.
# Variance: 0.000796757464004728


#OpenCell
# Total of 71520 images were sampled. (sample_size_per_markers=200)
# Variance: 0.007928811945021152
# Variance: 0.007928811945021152