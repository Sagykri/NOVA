
from multiprocessing import Pool
import numpy as np
import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from src.common.lib.image_sampling_utils import sample_images_all_markers_all_lines, sample_images_all_markers
from src.common.lib.preprocessing_utils import rescale_intensity

BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
INPUT_DIR = os.path.join(BASE_DIR, 'input', 'images', 'raw', 'SpinningDisk')

def check_batch(batch_name, sample_size_per_markers=100, num_markers=36):
    
    print(f"\n** check_batch {batch_name} **")
    
    # Global paths
    BATCH_TO_RUN = batch_name
    INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN)

    images = sample_images_all_markers_all_lines(INPUT_DIR_BATCH, 
                                                 sample_size_per_markers, 
                                                 num_markers,
                                                 raw=True,
                                                 rep_count=2, cond_count=2)
    
    results = _multiproc_check_batch(images_paths=images, batch_name=batch_name.replace("/", "_"))
    
    return results

def _multiproc_check_batch(images_paths, batch_name):
    
    images = images_paths
    n_images  = len(images)
    print("Total of", n_images, "images were sampled.")
    
    scores = []
    with Pool() as mp_pool:    
        
        for blur_score in mp_pool.map(_calc_image_blur, ([img_path for img_path in images])):
            if blur_score:
                scores.append(blur_score) 
        
        # if wish to run sequentially, and not multiproc
        # for img_path in images:
        #     blur_score = _calc_image_blur(img_path)
        #     scores.append(blur_score) 
        
        mp_pool.close()
        mp_pool.join()
    
    scores = np.array(scores)
    print("Plotting..")
    #plt.hist(scores, bins=10, color='black', fc='k', ec='k')
    sns.boxplot(data=scores)
    plt.title(f"{batch_name} preprocessing check_blur_images", c='purple')
    plt.savefig(f"src/preprocessing/{batch_name}_microglia_blur_scores.png")
    
    print(f"\nBlur scores - mean: {scores.mean()} std: {scores.std()}")
    return 

def _calc_image_blur(img_path):
    
    # Load an tiff image (a site image, 1024x1024)
    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH) 
    
    if img is None:
        path = img_path.replace("/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk/","")
        print(f"{path} is empty!")
        return None
    else:
        scaled_img = rescale_intensity(img)
        blur_value = np.sum(np.diff(scaled_img, axis=0)**2)
        return blur_value
    

if __name__ == '__main__':
    
    print("\n\n\nStart..")
    
    batches = [
            'batch7', #'batch6', 'batch9', 'batch8', 'batch3', 'batch4', 'batch5',
            #'deltaNLS_sort/batch2', 'deltaNLS_sort/batch3', 'deltaNLS_sort/batch4', 'deltaNLS_sort/batch5',
            #'NiemannPick_sort/batch1', 'NiemannPick_sort/batch2', 'NiemannPick_sort/batch3', 'NiemannPick_sort/batch4', 
            'microglia_sort/batch2', #'microglia_sort/batch3', 'microglia_sort/batch4',
            #'microglia_LPS_sort/batch1', 'microglia_LPS_sort/batch2',
            #'Perturbations'
            ]

    for batch_name in batches:
        check_batch(batch_name)
    
    
    
    print("\n\n\n\nDone!")
