# TODO: MOVE TO A DIFFERENT FILE/FOLDER + UTILIZE CONFIGURATION (SAGY WROTE THIS TODO)


#from multiprocessing import Pool
#from datetime import datetime
import matplotlib.pyplot as plt
from skimage import io
import cv2
## import czifile # for CZI 

from glob import glob 
import numpy as np
import pandas as pd
import logging
import pathlib
import random
import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')

from src.common.lib.image_sampling_utils import find_marker_folders


def test_raw_pixels_batch2_all(batch_num=2):
    
    print(f"\n\n test_raw_pixels_batch2_all - batch_num={batch_num}")
    
    # Global paths
    BATCH_TO_RUN = 'batch' + str(batch_num)
    INPUT_DIR = os.path.join(BASE_DIR, 'input','images','raw')
    INPUT_DIR_BATCH = os.path.join(INPUT_DIR, '220814_neurons')
    marker_folders = find_marker_folders(INPUT_DIR_BATCH, depth=3)
    
    values = []
    count = 0 
    for marker_files in marker_folders:
        print(marker_files)
        if marker_files[-3:]=="tif":
            site_image = io.imread(marker_files)
            # Get only marker channels
            site_image = site_image[:,:,[0,1,2]]
            # go over all three channels
            for c in range(0,3):
                target_image = site_image[:,:,c]
                values.extend(target_image.flatten())
                count+=1
    
    print(f"count={count}")
    plt.hist(np.array(values), bins=10)
    plt.savefig(os.path.join(BASE_DIR, 'outputs', 'pix_intensities_nancy_test', 'all_raw_pix_values_b' + str(batch_num) + '.png'))
    plt.close()


def test_normalized_pixels_all(batch_num):
    print(f"\n\n test_normalized_pixels - batch_num={batch_num}")
    # Global paths
    BATCH_TO_RUN = 'batch' + str(batch_num)

    BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
    INPUT_DIR = os.path.join(BASE_DIR,'input','images','processed','spd2','SpinningDisk')
    INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN)
    
    marker_folders = find_marker_folders(INPUT_DIR_BATCH, depth=3)
    
    for marker_folder in marker_folders:
        print(f"\n\n marker_folder={marker_folder}")
        input_data_dir = os.path.join(INPUT_DIR_BATCH, marker_folder)
        # Get npy file names under this batch
        images = sorted(os.listdir(input_data_dir))
        print("\n\nTotal of", len(images), "images were sampled.")
        
        values = []
        stats = []
        n_tiles = 0
        for processed_image_file_name in images:
            image_tiles = np.load(os.path.join(input_data_dir, processed_image_file_name))
            # Get only marker channel
            target_tiles = image_tiles[:,:,:,1]
            for i, tile in enumerate(target_tiles):
                
                values.extend(tile.flatten())
                n_tiles+=1
        print(f"number of tiles: {n_tiles}")
                
        plt.hist(np.array(values), bins=10)
        plt.xlim([0,1])
        plt.savefig(os.path.join(BASE_DIR, 'outputs', 'pix_intensities_nancy_test', 'preprocessed_pix_values_b' + str(batch_num) + '.png'))
        plt.close()
    return 

    
def test_normalized_pixels(batch_num):
    
    print(f"\n\n test_normalized_pixels - batch_num={batch_num}")
    # Global paths
    BATCH_TO_RUN = 'batch' + str(batch_num)

    BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
    INPUT_DIR = os.path.join(BASE_DIR,'input','images','processed','spd2','SpinningDisk')
    INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN)
    
    if batch_num==2:
        input_data_dir = os.path.join(INPUT_DIR_BATCH, 'WT','unstressed','PURA')
    else:
        input_data_dir = os.path.join(INPUT_DIR_BATCH, 'WT','Untreated','PURA')

    
    # Get npy file names under this batch
    images = sorted(os.listdir(input_data_dir))
    print("\n\nTotal of", len(images), "images were sampled.")
    
    values = []
    stats = []
    #processed_imgae_file_name = images[0]
    n_tiles = 0
    for processed_image_file_name in images:
        #marker_name = processed_image_file_name.split("/")[-2]
        image_tiles = np.load(os.path.join(input_data_dir, processed_image_file_name))
        # Get only marker channel
        target_tiles = image_tiles[:,:,:,1]
        for i, tile in enumerate(target_tiles):
            min, max, mean, std = tile.min(), tile.max(), tile.mean(), tile.std()
            span = max - min
            stats.append(span)
            values.extend(tile.flatten())
            n_tiles+=1
    print(f"number of tiles: {n_tiles}")
            
            
    # NOTE: two wells (rep)
    
    print("X")
    # plot histogram 
    plt.hist(stats, bins='auto')
    plt.xlim([0,1])
    plt.savefig(os.path.join(BASE_DIR, 'outputs', 'pix_intensities_nancy_test', 'preprocessed_span_b' + str(batch_num) + '.png'))
    plt.close()
    print("XX")
    plt.hist(np.array(values), bins=10)
    plt.xlim([0,1])
    plt.savefig(os.path.join(BASE_DIR, 'outputs', 'pix_intensities_nancy_test', 'preprocessed_pix_values_b' + str(batch_num) + '.png'))
    plt.close()
    print("XXX")
    return images

def test_raw_pixels(batch_num):
    
    print(f"\n\n test_raw_pixels - batch_num={batch_num}")
    
    # Global paths
    BATCH_TO_RUN = 'batch' + str(batch_num)

    INPUT_DIR = os.path.join(BASE_DIR, 'input','images','raw')
    
    if batch_num==2:
        INPUT_DIR_BATCH = os.path.join(INPUT_DIR, '220814_neurons', 'WT','unstressed')
        input_data_dir = os.path.join(INPUT_DIR_BATCH, '220808_iNDI_WT_unstressed-DAPI_PURA_SQSTM1_FMRP-01.tif')
        
        values = []
        stats = []
        site_image = io.imread(input_data_dir)
        # Get only marker channel
        pura_image = site_image[:,:,2]
        #cv2.imwrite(os.path.join(BASE_DIR, 'outputs','is_this_pura?.tif'), pura_image)
        
        min, max, mean, std = pura_image.min(), pura_image.max(), pura_image.mean(), pura_image.std()
        print(f"pura_image.shape:{pura_image.shape}, min:{min}, max:{max}, mean:{mean}, std:{std}")
        span = max - min
        stats.append(span)
        values.extend(pura_image.flatten())
        
    else:
        INPUT_DIR_BATCH = os.path.join(INPUT_DIR, 'SpinningDisk', BATCH_TO_RUN)
        input_data_dir = os.path.join(INPUT_DIR_BATCH, 'WT','panelA','Untreated', 'rep1', 'PURA')

        # Get tiff files names 
        site_images = sorted(os.listdir(input_data_dir))
        print(f"Loaded {len(site_images)} images from {input_data_dir}")
        
        values = []
        stats = []
        for site_image_file_name in site_images:
            site_image = io.imread(os.path.join(input_data_dir, site_image_file_name))
            min, max, mean, std = site_image.min(), site_image.max(), site_image.mean(), site_image.std()
            span = max - min
            print(f"min:{min}, max:{max}, mean:{mean}, std:{std}, span:{span}")            
            stats.append(span)
            values.extend(site_image.flatten())
            
    # plot histogram 
    b, bins, patches = plt.hist(stats, bins='auto')
    plt.savefig(os.path.join(BASE_DIR, 'outputs', 'pix_intensities_nancy_test', 'raw_span_b' + str(batch_num) + '.png'))

    print(len(values))
    print(f"pix values >20000 {(np.array(values) >20000).sum()}")

    plt.hist(np.array(values), bins=3)
    plt.savefig(os.path.join(BASE_DIR, 'outputs', 'pix_intensities_nancy_test', 'raw_pix_values_b' + str(batch_num) + '.png'))
    plt.close()


def test_normalized_pixels_all(batch_num):
    print(f"\n\n test_normalized_pixels - batch_num={batch_num}")
    # Global paths
    BATCH_TO_RUN = 'batch' + str(batch_num)

    BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
    INPUT_DIR = os.path.join(BASE_DIR,'input','images','processed','spd2','SpinningDisk')
    INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN)
    
    marker_folders = find_marker_folders(INPUT_DIR_BATCH, depth=3)
    
    for marker_folder in marker_folders:
        print(f"\n\n marker_folder={marker_folder}")
        input_data_dir = os.path.join(INPUT_DIR_BATCH, marker_folder)
        # Get npy file names under this batch
        images = sorted(os.listdir(input_data_dir))
        print("\n\nTotal of", len(images), "images were sampled.")
        
        values = []
        stats = []
        n_tiles = 0
        for processed_image_file_name in images:
            image_tiles = np.load(os.path.join(input_data_dir, processed_image_file_name))
            # Get only marker channel
            target_tiles = image_tiles[:,:,:,1]
            for i, tile in enumerate(target_tiles):
                
                values.extend(tile.flatten())
                n_tiles+=1
        print(f"number of tiles: {n_tiles}")
                
        plt.hist(np.array(values), bins=10)
        plt.xlim([0,1])
        plt.savefig(os.path.join(BASE_DIR, 'outputs', 'pix_intensities_nancy_test', 'preprocessed_pix_values_all_b' + str(batch_num) + '.png'))
        plt.close()
    return 

    
def test_normalized_pixels(batch_num):
    
    print(f"\n\n test_normalized_pixels - batch_num={batch_num}")
    # Global paths
    BATCH_TO_RUN = 'batch' + str(batch_num)

    BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
    INPUT_DIR = os.path.join(BASE_DIR,'input','images','processed','spd2','SpinningDisk')
    INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN)
    
    if batch_num==2:
        input_data_dir = os.path.join(INPUT_DIR_BATCH, 'WT','unstressed','PURA')
    else:
        input_data_dir = os.path.join(INPUT_DIR_BATCH, 'WT','Untreated','PURA')

    
    # Get npy file names under this batch
    images = sorted(os.listdir(input_data_dir))
    print("\n\nTotal of", len(images), "images were sampled.")
    
    values = []
    stats = []
    #processed_imgae_file_name = images[0]
    n_tiles = 0
    for processed_image_file_name in images:
        #marker_name = processed_image_file_name.split("/")[-2]
        image_tiles = np.load(os.path.join(input_data_dir, processed_image_file_name))
        # Get only marker channel
        target_tiles = image_tiles[:,:,:,1]
        for i, tile in enumerate(target_tiles):
            min, max, mean, std = tile.min(), tile.max(), tile.mean(), tile.std()
            span = max - min
            stats.append(span)
            values.extend(tile.flatten())
            n_tiles+=1
    print(f"number of tiles: {n_tiles}")
            
            
    # NOTE: two wells (rep)
    
    print("X")
    # plot histogram 
    plt.hist(stats, bins='auto')
    plt.xlim([0,1])
    plt.savefig(os.path.join(BASE_DIR, 'outputs', 'pix_intensities_nancy_test', 'preprocessed_span_b' + str(batch_num) + '.png'))
    plt.close()
    print("XX")
    plt.hist(np.array(values), bins=10)
    plt.xlim([0,1])
    plt.savefig(os.path.join(BASE_DIR, 'outputs', 'pix_intensities_nancy_test', 'preprocessed_pix_values_b' + str(batch_num) + '.png'))
    plt.close()
    print("XXX")
    return images

def test_raw_pixels(batch_num):
    
    print(f"\n\n test_raw_pixels - batch_num={batch_num}")
    
    # Global paths
    BATCH_TO_RUN = 'batch' + str(batch_num)

    INPUT_DIR = os.path.join(BASE_DIR, 'input','images','raw')
    
    if batch_num==2:
        INPUT_DIR_BATCH = os.path.join(INPUT_DIR, '220814_neurons', 'WT','unstressed')
        input_data_dir = os.path.join(INPUT_DIR_BATCH, '220808_iNDI_WT_unstressed-DAPI_PURA_SQSTM1_FMRP-01.tif')
        
        values = []
        stats = []
        site_image = io.imread(input_data_dir)
        # Get only marker channel
        pura_image = site_image[:,:,2]
        #cv2.imwrite(os.path.join(BASE_DIR, 'outputs','is_this_pura?.tif'), pura_image)
        
        min, max, mean, std = pura_image.min(), pura_image.max(), pura_image.mean(), pura_image.std()
        print(f"pura_image.shape:{pura_image.shape}, min:{min}, max:{max}, mean:{mean}, std:{std}")
        span = max - min
        stats.append(span)
        values.extend(pura_image.flatten())
        
    else:
        INPUT_DIR_BATCH = os.path.join(INPUT_DIR, 'SpinningDisk', BATCH_TO_RUN)
        input_data_dir = os.path.join(INPUT_DIR_BATCH, 'WT','panelA','Untreated', 'rep1', 'PURA')

        # Get tiff files names 
        site_images = sorted(os.listdir(input_data_dir))
        print(f"Loaded {len(site_images)} images from {input_data_dir}")
        
        values = []
        stats = []
        for site_image_file_name in site_images:
            site_image = io.imread(os.path.join(input_data_dir, site_image_file_name))
            min, max, mean, std = site_image.min(), site_image.max(), site_image.mean(), site_image.std()
            span = max - min
            print(f"min:{min}, max:{max}, mean:{mean}, std:{std}, span:{span}")            
            stats.append(span)
            values.extend(site_image.flatten())
            
    # plot histogram 
    b, bins, patches = plt.hist(stats, bins='auto')
    plt.savefig(os.path.join(BASE_DIR, 'outputs', 'pix_intensities_nancy_test', 'raw_span_b' + str(batch_num) + '.png'))

    print(len(values))
    print(f"pix values >20000 {(np.array(values) >20000).sum()}")

    plt.hist(np.array(values), bins=3)
    plt.savefig(os.path.join(BASE_DIR, 'outputs', 'pix_intensities_nancy_test', 'raw_pix_values_b' + str(batch_num) + '.png'))
    plt.close()


def test_normalized_pixels_all(batch_num):
    print(f"\n\n test_normalized_pixels - batch_num={batch_num}")
    # Global paths
    BATCH_TO_RUN = 'batch' + str(batch_num)

    BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
    INPUT_DIR = os.path.join(BASE_DIR,'input','images','processed','spd2','SpinningDisk')
    INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN)
    
    marker_folders = find_marker_folders(INPUT_DIR_BATCH, depth=3)
    
    for marker_folder in marker_folders:
        print(f"\n\n marker_folder={marker_folder}")
        input_data_dir = os.path.join(INPUT_DIR_BATCH, marker_folder)
        # Get npy file names under this batch
        images = sorted(os.listdir(input_data_dir))
        print("\n\nTotal of", len(images), "images were sampled.")
        
        values = []
        stats = []
        n_tiles = 0
        for processed_image_file_name in images:
            image_tiles = np.load(os.path.join(input_data_dir, processed_image_file_name))
            # Get only marker channel
            target_tiles = image_tiles[:,:,:,1]
            for i, tile in enumerate(target_tiles):
                
                values.extend(tile.flatten())
                n_tiles+=1
        print(f"number of tiles: {n_tiles}")
                
        plt.hist(np.array(values), bins=10)
        plt.xlim([0,1])
        plt.savefig(os.path.join(BASE_DIR, 'outputs', 'pix_intensities_nancy_test', 'preprocessed_pix_values_b' + str(batch_num) + '.png'))
        plt.close()
    return 

    
def test_normalized_pixels(batch_num):
    
    print(f"\n\n test_normalized_pixels - batch_num={batch_num}")
    # Global paths
    BATCH_TO_RUN = 'batch' + str(batch_num)

    BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
    INPUT_DIR = os.path.join(BASE_DIR,'input','images','processed','spd2','SpinningDisk')
    INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN)
    
    if batch_num==2:
        input_data_dir = os.path.join(INPUT_DIR_BATCH, 'WT','unstressed','PURA')
    else:
        input_data_dir = os.path.join(INPUT_DIR_BATCH, 'WT','Untreated','PURA')

    
    # Get npy file names under this batch
    images = sorted(os.listdir(input_data_dir))
    print("\n\nTotal of", len(images), "images were sampled.")
    
    values = []
    stats = []
    #processed_imgae_file_name = images[0]
    n_tiles = 0
    for processed_image_file_name in images:
        #marker_name = processed_image_file_name.split("/")[-2]
        image_tiles = np.load(os.path.join(input_data_dir, processed_image_file_name))
        # Get only marker channel
        target_tiles = image_tiles[:,:,:,1]
        for i, tile in enumerate(target_tiles):
            min, max, mean, std = tile.min(), tile.max(), tile.mean(), tile.std()
            span = max - min
            stats.append(span)
            values.extend(tile.flatten())
            n_tiles+=1
    print(f"number of tiles: {n_tiles}")
            
            
    # NOTE: two wells (rep)
    
    print("X")
    # plot histogram 
    plt.hist(stats, bins='auto')
    plt.xlim([0,1])
    plt.savefig(os.path.join(BASE_DIR, 'outputs', 'pix_intensities_nancy_test', 'preprocessed_span_b' + str(batch_num) + '.png'))
    plt.close()
    print("XX")
    plt.hist(np.array(values), bins=10)
    plt.xlim([0,1])
    plt.savefig(os.path.join(BASE_DIR, 'outputs', 'pix_intensities_nancy_test', 'preprocessed_pix_values_b' + str(batch_num) + '.png'))
    plt.close()
    print("XXX")
    return images

def test_raw_pixels(batch_num):
    
    print(f"\n\n test_raw_pixels - batch_num={batch_num}")
    
    # Global paths
    BATCH_TO_RUN = 'batch' + str(batch_num)

    INPUT_DIR = os.path.join(BASE_DIR, 'input','images','raw')
    
    if batch_num==2:
        INPUT_DIR_BATCH = os.path.join(INPUT_DIR, '220814_neurons', 'WT','unstressed')
        input_data_dir = os.path.join(INPUT_DIR_BATCH, '220808_iNDI_WT_unstressed-DAPI_PURA_SQSTM1_FMRP-01.tif')
        
        values = []
        stats = []
        site_image = io.imread(input_data_dir)
        # Get only marker channel
        pura_image = site_image[:,:,2]
        #cv2.imwrite(os.path.join(BASE_DIR, 'outputs','is_this_pura?.tif'), pura_image)
        
        min, max, mean, std = pura_image.min(), pura_image.max(), pura_image.mean(), pura_image.std()
        print(f"pura_image.shape:{pura_image.shape}, min:{min}, max:{max}, mean:{mean}, std:{std}")
        span = max - min
        stats.append(span)
        values.extend(pura_image.flatten())
        
    else:
        INPUT_DIR_BATCH = os.path.join(INPUT_DIR, 'SpinningDisk', BATCH_TO_RUN)
        input_data_dir = os.path.join(INPUT_DIR_BATCH, 'WT','panelA','Untreated', 'rep1', 'PURA')

        # Get tiff files names 
        site_images = sorted(os.listdir(input_data_dir))
        print(f"Loaded {len(site_images)} images from {input_data_dir}")
        
        values = []
        stats = []
        for site_image_file_name in site_images:
            site_image = io.imread(os.path.join(input_data_dir, site_image_file_name))
            min, max, mean, std = site_image.min(), site_image.max(), site_image.mean(), site_image.std()
            span = max - min
            print(f"min:{min}, max:{max}, mean:{mean}, std:{std}, span:{span}")            
            stats.append(span)
            values.extend(site_image.flatten())
            
    # plot histogram 
    b, bins, patches = plt.hist(stats, bins='auto')
    plt.savefig(os.path.join(BASE_DIR, 'outputs', 'pix_intensities_nancy_test', 'raw_span_b' + str(batch_num) + '.png'))

    print(len(values))
    print(f"pix values >20000 {(np.array(values) >20000).sum()}")

    plt.hist(np.array(values), bins=3)
    plt.savefig(os.path.join(BASE_DIR, 'outputs', 'pix_intensities_nancy_test', 'raw_pix_values_b' + str(batch_num) + '.png'))
    plt.close()
    
def test_czi_pix_scale():
     
     DATA_DIR = '/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/tests/images_for_testing/'
     czi_file_path = os.path.join(DATA_DIR, 'czi', '220805_iNDI_WT_unstressed_A-01.czi')
     
     # load czi image from confocal microscope
     img_data = czifile.imread(czi_file_path)
     print(f"czi_file: {czi_file_path} dimension:{img_data.shape}")
     # czi files dimension are: slices, time series, scenes, channels, x, y, z, RGB
     # (1, 1, 1, 4, 1, 13414, 13414, 1)
     img_data = img_data[0,0,0,:,0,:,:,0]
     print(f"extract relevant dimension:{img_data.shape}")
     for i in range(0,4):
         c_data = img_data[i,:,:]
         print(f"channel {i}: shape: {c_data.shape} min: {c_data.min()} min: {c_data.min()} max: {c_data.max()} mean: {c_data.mean()} std: {c_data.std()} span: {c_data.max()-c_data.min()}")
    #img_data[0,:,:] = img_data[0,:,:] / 65535
     return None
 
def test_fiji_pix_scale():
     
     DATA_DIR = '/home/labs/hornsteinlab/Collaboration/MOmaps_Sagy/MOmaps/tests/images_for_testing/fiji'
     
     
     for file in sorted(os.listdir(os.path.join(DATA_DIR, 'raw'))):         
         pre_fiji_img = io.imread(os.path.join(DATA_DIR, 'raw', file))
         post_fiji_img = io.imread(os.path.join(DATA_DIR, 'converted', file))
         print(f"pre_fiji_img {file}: shape: {pre_fiji_img.shape} min: {pre_fiji_img.min()} min: {pre_fiji_img.min()} max: {pre_fiji_img.max()} mean: {pre_fiji_img.mean()} std: {pre_fiji_img.std()} span: {pre_fiji_img.max()-pre_fiji_img.min()}")
         print(f"post_fiji_img {file}: shape: {post_fiji_img.shape} min: {post_fiji_img.min()} min: {post_fiji_img.min()} max: {post_fiji_img.max()} mean: {post_fiji_img.mean()} std: {post_fiji_img.std()} span: {post_fiji_img.max()-post_fiji_img.min()}")

     return None

def test_data_normalization_vs_contrast_normalization():
    
    BATCH_TO_RUN = 'batch8'
    INPUT_DIR_BATCH = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps', 'input','images','raw','SpinningDisk', BATCH_TO_RUN)
    input_data_dir = os.path.join(INPUT_DIR_BATCH, 'WT','panelC','Untreated', 'rep1', 'SQSTM1')

    # Get tiff files names 
    site_images = sorted(os.listdir(input_data_dir))
    print(f"Loaded {len(site_images)} images from {input_data_dir}")
    
    values = []
    stats = []
    for site_image_file_name in site_images:
        site_image = io.imread(os.path.join(input_data_dir, site_image_file_name))
        min, max, mean, std = site_image.min(), site_image.max(), site_image.mean(), site_image.std()
        span = max - min
        print(f"min:{min}, max:{max}, mean:{mean}, std:{std}, span:{span}") 
        # save orig site_image
        cv2.imwrite(os.path.join(BASE_DIR, 'outputs', 'pix_intensities_nancy_test', 'SQSTM1_to8bit', 'orig_' + BATCH_TO_RUN +'_'+ site_image_file_name +'.png'), 
                    site_image)
        # save scaled site_image
        img_scaled = cv2.normalize(site_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) 
        cv2.imwrite(os.path.join(BASE_DIR, 'outputs', 'pix_intensities_nancy_test', 'SQSTM1_to8bit', '255_scaled_' + BATCH_TO_RUN +'_'+ site_image_file_name +'.png'), 
                    img_scaled)
        

        min, max, mean, std = img_scaled.min(), img_scaled.max(), img_scaled.mean(), img_scaled.std()
        span = max - min
        print(f"img_scaled - min:{min}, max:{max}, mean:{mean}, std:{std}, span:{span}") 
        stats.append(span)
        values.extend(site_image.flatten())
        
    # plot histogram 
    plt.hist(stats, bins='auto')
    plt.savefig(os.path.join(BASE_DIR, 'outputs', 'pix_intensities_nancy_test', '255_scaled_span_' + BATCH_TO_RUN + '.png'))

    #print(len(values))
    #print(f"pix values >20000 {(np.array(values) >20000).sum()}")

    plt.hist(np.array(values), bins=3)
    plt.savefig(os.path.join(BASE_DIR, 'outputs', 'pix_intensities_nancy_test', '255_scaled_pix_values_' + BATCH_TO_RUN + '.png'))
    plt.close()

    
    
    # Using the contrast_normalization formula with NumPy and to remap data back to the 8-bit input format after contrast normalization, the image should be multiplied by max intensity, then change the data type back to unit8:
    # cn_image_correct = (c_image - c_image.min()) / (c_image.max()- c_image.min()) * 255
    # cn_image_correct = cn_image_correct.astype(np.int8)
if __name__ == '__main__':
    
    print("\n\n\nStart..")
    
    #test_normalized_pixels(batch_num=2)
    #test_normalized_pixels(batch_num=8)
    test_raw_pixels(batch_num=8)
    #test_raw_pixels(batch_num=2)
    #test_czi_pix_scale()
    #test_fiji_pix_scale()
    
    #test_data_normalization_vs_contrast_normalization()
    
    #test_normalized_pixels_all(batch_num=2)
    #test_raw_pixels_batch2_all()
    
    print("\n\n\n\nDone!")
    
    