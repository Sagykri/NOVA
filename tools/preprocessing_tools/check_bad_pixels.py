##########################################
# Checking all datasets all batches (in multiprocessing) 
# for pixels with value of 0 or 1
##########################################
from multiprocessing import Pool
import numpy as np
import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import cv2
from src.common.lib.image_sampling_utils import sample_images_all_markers_all_lines, sample_images_all_markers

BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
INPUT_DIR = os.path.join(BASE_DIR, 'input', 'images', 'raw', 'SpinningDisk')

def check_batch(batch_name, sample_size_per_markers=200, num_markers=36):
    
    print(f"\n** check_batch {batch_name} **")
    
    # Global paths
    BATCH_TO_RUN = batch_name
    INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN)

    images = sample_images_all_markers_all_lines(INPUT_DIR_BATCH, 
                                                 sample_size_per_markers, 
                                                 num_markers,
                                                 raw=True, all_conds=True,
                                                 rep_count=2, cond_count=2, exclude_DAPI=False)
    
    results = _multiproc_check_batch(images_paths=images)
    
    return results

def _multiproc_check_batch(images_paths):
    
    images = images_paths
    n_images  = len(images)
    print("Total of", n_images, "images were sampled.")
    
    vars = []
    counts = {}
    with Pool() as mp_pool:    
        
        for image_name in mp_pool.map(_check_image, ([img_path for img_path in images])):
            if image_name and image_name not in vars:
                vars.append(image_name) 
        
            if image_name in counts:
                counts[image_name] += 1
            else:
                counts[image_name] = 1
        
        # if wish to run sequentially, and not multiproc
        # for img_path in images:
        #     image_name = _check_image(img_path)
        #     vars.append(image_name) 
        
        mp_pool.close()
        mp_pool.join()
    
    print(f"\Groups of files with different bad pixels: {len(vars)}")
    print(f"\n{counts}")
    return 

def _check_image(img_path):
    
    # Load an tiff image (a site image, 1024x1024)
    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH) 
    
    if img is None:
        print(f"{img_path} is empty!")
        return None
    else:
        zeros = np.where(img.reshape(-1,)==0)[0]
        rows = zeros//img.shape[0]
        cols = zeros%img.shape[1]
        zeros_locs = list(zip(rows, cols))
        n_zeros = len(zeros_locs)
        
        ones = np.where(img.reshape(-1,)==1)[0]
        rows = ones//img.shape[0]
        cols = ones%img.shape[1]
        ones_locs = list(zip(rows, cols))
        n_ones = len(ones_locs)
        
        #print(n_zeros, zeros_locs)
        #print(n_ones, ones_locs)
        if (n_zeros!=3) or \
            (zeros_locs!=[(1023, 1021), (1023, 1022), (1023, 1023)]) or \
                (n_ones!=1) or \
                    (ones_locs!=[(1023, 1020)]):
            img_path = img_path.replace("/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk/","")
            res = (img_path.split("/")[-2], img_path.split("/")[-3], img_path.split("/")[-4], img_path.split("/")[-5], img_path.split("/")[-6])
            return res
    

if __name__ == '__main__':
    
    print("\n\n\nStart..")
    
    batches = [
            #'batch7', 'batch6', 'batch9', 'batch8', 'batch3', 'batch4', 'batch5',
            #'deltaNLS_sort/batch2', 'deltaNLS_sort/batch3', 'deltaNLS_sort/batch4', 'deltaNLS_sort/batch5',
            #'NiemannPick_sort/batch1', 'NiemannPick_sort/batch2', 'NiemannPick_sort/batch3', 'NiemannPick_sort/batch4', 
            #'microglia_sort/batch2', 'microglia_sort/batch3', 'microglia_sort/batch4',
            'microglia_LPS_sort/batch1', #'microglia_LPS_sort/batch2',
            #'Perturbations'
            ]

    for batch_name in batches:
        check_batch(batch_name)
    
    
    
    print("\n\n\n\nDone!")
