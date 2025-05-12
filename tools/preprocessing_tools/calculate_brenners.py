from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import sklearn
import os
import os
import numpy as np
import concurrent.futures
from skimage.feature import hog
from skimage import exposure
from skimage.feature import local_binary_pattern
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from PIL import Image
import sys
import logging

sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

from src.preprocessing.preprocessing_utils import fit_image_shape, get_image_focus_quality, rescale_intensity
from src.common.utils import init_logging, flat_list_of_lists
from tools.preprocessing_tools.image_sampling_utils import sample_images_all_markers_all_lines


BASE_DIR = os.path.join('/home','projects','hornsteinlab','Collaboration','MOmaps')
INPUT_DIR = os.path.join(BASE_DIR, 'input', 'images', 'raw', 'SpinningDisk', 'NOVA_d18_neurons_sorted')
calc_per_tile = False # I ran _site_ with this being False! (281123)

def calculate_metrics_for_batch(batch_name, sample_size_per_markers=100, num_markers=36, markers=None):
    
    logging.info(f"\n** check_batch {batch_name} **")
    
    # Global paths
    BATCH_TO_RUN = batch_name
    INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN)

    images = sample_images_all_markers_all_lines(INPUT_DIR_BATCH, 
                                                 sample_size_per_markers, 
                                                 num_markers,
                                                 raw=True,
                                                 rep_count=2,
                                                #  cond_count=2,
                                                 all_conds=True)
                                                #  markers=["DAPI"])
    logging.info(f"Images len: {len(images)}")
    if markers is not None:
        images = [img for img in images if img.split(os.sep)[-2] in markers]
        logging.info(f"Images after fitlering by markers len: {len(images)} (Markers: {markers})")
    results = _multiproc_calcualte_metrics_for_batch(images_paths=images)
    
    return results

def split_array(input_array, tile_h=100, tile_w=100):
    height, width = input_array.shape
    sub_arrays = []

    for y in range(0, height, tile_h):
        for x in range(0, width, tile_w):
            sub_array = input_array[y:y+tile_h, x:x+tile_w]
            sub_arrays.append(sub_array)
    
    return np.asarray(sub_arrays)

def _multiproc_calcualte_metrics_for_batch(images_paths):
    
    n_images  = len(images_paths)
    logging.info(f"Total of {n_images} images were sampled.")
    
    
    with Pool() as mp_pool:    
        results = mp_pool.map(_calc_image_metrics, (images_paths))
        
        mp_pool.close()
        mp_pool.join()
    
    if calc_per_tile:
        results = flat_list_of_lists(results)
    
    return results
    
def _calc_image_metrics(img_path):
    logging.info(img_path)
    # Load an tiff image (a site image, 1024x1024)
    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH) 
    if img is None:
        path = img_path.replace("/home/projects/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk/NOVA_d18_neurons_sorted/","")
        logging.info(f"{path} is empty!")
        return None
    
    # img = img[12:-12,12:-12]
    img = fit_image_shape(img, (1024, 1024))
    
    scaled_img = rescale_intensity(img)
    
    if not calc_per_tile:
        return (img_path, ) + get_metrics(scaled_img)
    
    tiles = split_array(scaled_img)
    
    ret = []
    for i, t in enumerate(tiles):
        row = (f"{img_path}_{i}", ) + get_metrics(t)
        ret.append(row)
        
    return ret

def save_to_file(results, savepath):
    # To Dataframe
    logging.info("To dataframe")
    columns = ["Path", 'Target_Sharpness_Brenner']
    

    df = pd.DataFrame(results, columns=columns)
    df.insert(1, 'Batch', df.apply(lambda x: x['Path'].split(os.sep)[-7], axis=1))
    df.insert(1, 'Rep', df.apply(lambda x: x['Path'].split(os.sep)[-3].split('_',1)[0], axis=1))
    df.insert(1, 'Batch_Rep', df.apply(lambda x: '/'.join([x['Batch'], x['Rep']]), axis=1))
    df.insert(1, 'CellLine', df.apply(lambda x: x['Path'].split(os.sep)[-6], axis=1))
    df.insert(1, 'Condition', df.apply(lambda x: x['Path'].split(os.sep)[-4], axis=1))
    df.insert(1, 'Marker', df.apply(lambda x: x['Path'].split(os.sep)[-2], axis=1))
    df.insert(1, 'RootFolder', df.apply(lambda x: os.sep.join(x['Path'].split(os.sep)[:-1]), axis=1))
    logging.info(f"Saving df to {savepath}")
    df.to_csv(savepath, index=False)

def get_metrics(tile):
    sharpness_brenner = get_image_focus_quality(tile)    
    return (sharpness_brenner,)


def main():
    # cell_lines = ['WT']
    # conditions = ['Untreated']#, 'stress']
    # markers = #['DAPI']#["DAPI"]#['NONO', 'G3BP1']
    batches = ['batch1', 'batch2']#[os.path.join('240323_day29_neurons_sorted', 'batch1')]#['batch4','batch5','batch6', 'batch9']#['batch7', 'batch8', 'batch3', 'batch4','batch5','batch6', 'batch9']#, 'batch8']#['batch6_16bit_no_downsample']
    # raw_base_path = '/home/projects/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk/'
    
    
    log_file_path = "/home/projects/hornsteinlab/Collaboration/MOmaps/outputs/preprocessing/spd18days/brenner/log280524_all.txt"
    savepath =      "/home/projects/hornsteinlab/Collaboration/MOmaps/outputs/preprocessing/spd18days/brenner/raw_metrics280524_all.csv"
    
    init_logging(log_file_path)
    
    logging.info("Starting outlier detection..")
    
    results = []
    
    for batch_name in batches:
        logging.info(f"Calculating metrics for batch: {batch_name}")
        results_batch = calculate_metrics_for_batch(batch_name)
        logging.info(f"Appending metrics from batch {batch_name}")
        results.extend(results_batch)
        save_to_file(results, f"{savepath}_checkpoint_{batch_name.replace(os.sep, '.')}")
    
    save_to_file(results, savepath)
    
if __name__ == "__main__":
    print("Starting outlier detection...")
    try:
        main()
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
    
    
###########################################


