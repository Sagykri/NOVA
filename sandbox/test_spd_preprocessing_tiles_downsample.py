from multiprocessing import Pool
from datetime import datetime
from glob import glob 
import logging
import pathlib
import random
import os
import sys


import numpy as np
from skimage import io, transform
from skimage.measure import block_reduce
import matplotlib.pyplot as plt

import sys
#sys.path.insert(1, os.getenv("MOMAPS_HOME")) #TODO: this doesn't work for me. MOMAPS_HOME is set in my linux but get() returns empty
sys.path.insert(1,'/home/labs/hornsteinlab/Collaboration/MOmaps/')
from src.common.lib import preprocessing_utils

# Global paths
BATCH_TO_RUN = 'batch8' 

BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
INPUT_DIR = os.path.join(BASE_DIR,'input','images','raw','SpinningDisk')
INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN)


def find_marker_folders(batch_path, depth=5):
    """Returns paths of all marker folder in a batch
    Args:
        batch_path (string):  full path of batch folder
        depth (int, optional): depth of marker sub-folders. Defaults to 5.
        Note: Markers are assumend to be always in a given constant "depth"

    Yields:
        string: a path of marker folder
    """
    
    # Recursively list files and directories up to a certain depth
    depth -= 1
    with os.scandir(batch_path) as input_data_folder:
        for entry in input_data_folder:
            
            # if that's not a marker directory, recursion...
            if entry.is_dir() and depth > 0:
                yield from find_marker_folders(entry.path, depth)
            
            # if that's a marker directory
            elif depth==0: 
                marker_name = os.path.basename(entry.path)
                if marker_name=='DAPI':
                    continue
                else:
                    # This is a list of arguments, used as the input of analyze_marker()
                    yield entry.path

def sample_image_names_per_marker(input_data_dir, sample_size=1):
    """
    For a given marker, this function samples file names of images of target marker and DAPI marker 
    
    Args:
        input_data_dir (string): full path of marker directory
        Note: "input_data_dir" has to point to a marker directory
        sample_size (int, optional): how  many images to sample. Defaults to 1.

    Returns:
        _type_: _description_
    """
    
    
    print(f"\nsample_image_names_per_marker: {input_data_dir}. {sample_size} images per marker.")
    
    # This will hold the full path of n images (n is defined by "sample_size") of the marker
    print(os.listdir(input_data_dir), len(os.listdir(input_data_dir)), sample_size)
    filenames = random.sample(os.listdir(input_data_dir), sample_size)
    # get the assocoated DAPI folder 
    nucleus_folder = os.path.join(pathlib.Path(input_data_dir).parent.resolve(), "DAPI")
    
    files_list = []
    # Target marker
    for target_file in filenames:
        filename, ext = os.path.splitext(target_file)
        if ext == '.tif':
            image_filename = os.path.join(input_data_dir, target_file)
        
            
            # Find the assocoated  Nucli marker (DAPI) file of this target image 
            # target and DAPI should have same sufix (site it) 
            site = filename.split('_')[-1]
            nucleus_filepath = glob(f"{nucleus_folder}/*_{site}{ext}")
            if len(nucleus_filepath) == 0:
                print(f"Skipping site {site} for {target_file} since no DAPI for this site was found")
                continue
            
            # Add to list
            files_list.append({'target':image_filename, 'dapi':nucleus_filepath[0]})
        
        else:
            print(f"sampled file {target_file} was not a tif. re-sampling.. ")
            #sample_image_names_per_marker(input_data_dir, sample_size)
            continue

    print("\n\n\nThe files sampled", files_list)
            
    return files_list

def sample_images_all_markers(sample_size_per_markers=1, num_markers=10):
    """Samples random raw images for a given batch 

    Args:
        sample_size_per_markers (int, optional): how many images to sample for each marker. Defaults to 1.
        num_markers (int, optional): how many markers to sample. Defaults to 10.

    Returns:
        list: list of paths (strings) 
    """
    
    sampled_images = []
    sampled_markers = set()
    
    # Get a list of all marker folders
    marker_subfolder = find_marker_folders(INPUT_DIR_BATCH, depth=5)
    # Sample n markers, and for each marker, sample k images (where n=num_markers and k=sample_size_per_markers)
    for marker_folder in marker_subfolder:
        n_images = 0
        
        if (len(sampled_markers) < num_markers):
        
            # exclude those cell lines, not very dense. Lena M
            if (n_images<sample_size_per_markers) and ('SCNA' not in marker_folder) and ('FUS' not in marker_folder):
        
                sampled_marker_images = sample_image_names_per_marker(marker_folder, sample_size=sample_size_per_markers)
                
                if sampled_marker_images:
                    sampled_images.extend(sampled_marker_images)
                    sampled_markers.add(marker_folder)
                    
                    n_images += 1
            if (n_images==sample_size_per_markers): 
                continue
        
    print("sampled_images:", len(sampled_images), "sampled_markers:", len(sampled_markers))
    return sampled_images

if False:
    def crop_to_tiles(tile_w, tile_h, img_processed):
        """Crop tiles to given size"""
        
        image_processed_tiles = []
        image_dim = len(img_processed.shape)
        image_w = img_processed.shape[0]
        image_h = img_processed.shape[1]
    
        to_validate = True
        if image_w % tile_w != 0 or image_h % tile_h != 0:
            to_validate = False
            logging.warning(f"[Warning] Fuzzy divizion ({tile_w}, {tile_h}; {image_w}, {image_h})")

        img_processed = img_processed[:image_w - image_w % tile_w, :image_h - image_h % tile_h]
        image_w = img_processed.shape[0]
        image_h = img_processed.shape[1]
        logging.info(f"[INFO] shape: {img_processed.shape}")

        if to_validate:
            n_tiles_expected = (image_w * image_w) // (tile_w * tile_h)

        for w in range(0, image_w, tile_w):
            for h in range(0, image_h, tile_h):
                image_processed_tiles.append(img_processed[w:w+tile_w, h:h+tile_h, :])

        image_processed_tiles = np.stack(image_processed_tiles, axis=image_dim)
        image_processed_tiles = np.moveaxis(image_processed_tiles, -1, 0)

        if to_validate and n_tiles_expected != image_processed_tiles.shape[0]:
            raise f"Error: #Expected tiles ({n_tiles_expected}) != #Observer tiles ({image_processed_tiles.shape[0]})"

        logging.info(f"Tiles shape {image_processed_tiles.shape}")
        return image_processed_tiles

    def downsample(show, img_current_channel, block_size):
        """Downsampling given image by 2"""
        
        img_current_channel = block_reduce(image=img_current_channel, block_size=block_size, func=np.mean)
        
        if show:
            plt.imshow(img_current_channel)
            plt.show()
        return img_current_channel

    def normalize(show, img_current_channel):
        """Normalize (min-max) given image"""
        
        image_test_min = np.min(img_current_channel)
        image_test_max = np.max(img_current_channel)
        img_current_channel -= image_test_min
        img_current_channel /= (image_test_max - image_test_min)
        logging.info(img_current_channel.shape)
        if show:  
            plt.imshow(img_current_channel)
            plt.show()
        
        return img_current_channel

def save_tile(img, save_path):
    
    # Increase contrast with min-max scaling
    #img = (img - img.min(axis=0)) / (img.max(axis=0) - img.min(axis=0)) # TODO: talk to Sagy about this, with this you see horizental lines
    
    # save image to "save_path" (full path + file name)
    plt.imsave(save_path, img, cmap = 'rainbow')
    return None

def preprocess_image(file):

    target_file_path = file['target']
    nucleus_file_path = file['dapi']
    
    #target_file_path = '/home/labs/hornsteinlab/Collaboration/MOmaps/data/raw/SpinningDisk/batch3/TDP43/panelE/Untreated/rep2/NEMO/R11_w3confCy5_s277.tif'
    #nucleus_file_path = '/home/labs/hornsteinlab/Collaboration/MOmaps/data/raw/SpinningDisk/batch3/TDP43/panelE/Untreated/rep2/DAPI/R11_w1confDAPI_s277.tif'

    img_target = io.imread(target_file_path)
    img_nucleus = io.imread(nucleus_file_path)

    # Take nuclues and target channels so target is the first channel and nuclues is the second
    img = np.stack([img_target, img_nucleus], axis=2)

    n_channels = img.shape[-1]

    img_processed = None
    for c in range(n_channels):
        img_current_channel = np.array(img[...,c], dtype=np.float64)
        if img_processed is None:
            img_processed = img_current_channel
        else:
            img_processed = np.dstack((img_processed, img_current_channel))
        
    return img_processed, target_file_path

def main(_tiles_size = 256, sample_size_per_markers=1, num_markers=10):
    
    # Sample markers and then sample images of these markers. The returened value is a list of paths (strings) 
    ##files = sample_images_all_markers(sample_size_per_markers, num_markers)
    
    files = [
        {
            'target': '/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk/batch8/TDP43/panelA/Untreated/rep1/G3BP1/R11_w3confCy5_s491.tif', 
            'dapi':   '/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk/batch8/TDP43/panelA/Untreated/rep1/DAPI/R11_w1confDAPI_s491.tif'
        },
        {
            'target': '/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk/batch8/TDP43/panelA/Untreated/rep1/G3BP1/R11_w3confCy5_s492.tif', 
            'dapi':   '/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk/batch8/TDP43/panelA/Untreated/rep1/DAPI/R11_w1confDAPI_s492.tif'
        },
        {
            'target': '/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk/batch8/TDP43/panelA/Untreated/rep1/KIF5A/R11_w2confmCherry_s492.tif', 
            'dapi':   '/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk/batch8/TDP43/panelA/Untreated/rep1/DAPI/R11_w1confDAPI_s492.tif'
        }
    ]
    
    for file_path in files:
        
        img_processed, orig_target_file_path = preprocess_image(file_path)

        # normalize original raw image
        preprocessing_utils.normalize(show=False, img_current_channel=img_processed)
        
        # crop the 1024x1024 image to 16 tiles, each of 256x256
        tiles = preprocessing_utils.crop_to_tiles(tile_w=_tiles_size, tile_h=_tiles_size, img_processed=img_processed)
        
        # name for new tile files
        tmp = orig_target_file_path.split('/input/images/raw/SpinningDisk/'+BATCH_TO_RUN, maxsplit=2)
        print(tmp)
        save_name = tmp[1].replace('/','_').replace('tif','')
        save_path = os.path.join(BASE_DIR,'sandbox','test_spd_tiles_for_Lena', save_name)

        size = 100
        _downsample_block_size=2
        for i, tile in enumerate(tiles):
            # get the target channel 
            target_tile = np.array(tile[...,0], dtype=np.float64)
            # downsample from 256x256 to 128x128
            ds_tile = preprocessing_utils.downsample(show=False, 
                                 img_current_channel=target_tile, 
                                 block_size=_downsample_block_size) #TODO: Sagy does "//" which returns int, but I need 2.56 no 2
            # resize to 100x100
            resized_tile = transform.resize(ds_tile, (size, size), anti_aliasing=True)
            
            # save the target tile to a png file
            final_path = save_path + "_target_tile_"+str(i)+".png"
            save_tile(resized_tile, final_path)
            print("save", final_path)
        
        for i, tile in enumerate(tiles):
            fig = plt.figure()
            # get the DAPI channel 
            dapi_tile = np.array(tile[...,1], dtype=np.float64)
            # downsample from 256x256 to 128x128
            ds_tile = preprocessing_utils.downsample(show=False, 
                                 img_current_channel=dapi_tile, 
                                 block_size=_downsample_block_size)
            
            #print("BEFORE:", ds_tile)
            #ds_tile_flt = ds_tile.flatten()
            #plt.scatter(np.arange(len(ds_tile_flt)), ds_tile_flt, alpha=0.5)
            
            # resize to 100x100
            resized_tile = transform.resize(ds_tile, (size, size), anti_aliasing=True)
            #print("AFTER:", resized_tile)
            #resized_tile_flt = resized_tile.flatten()
            #plt.scatter(np.arange(len(resized_tile_flt)), resized_tile_flt, alpha=0.5)
            # plt.show()
            # save the DAPI tile to a png file
            #final_path = save_path + "scatter_dapi_tile_"+str(i)+".png"
            #plt.savefig(final_path)
            #plt.close()
            final_path = save_path + "_DAPI_tile_"+str(i)+".png"
            
            save_tile(resized_tile, final_path)
            print("save", final_path)

    return None        

if __name__ == '__main__':
    print("\n\n\n\n\nStart..")
    main(_tiles_size = 256, sample_size_per_markers=5, num_markers=3)
    print("\n\n\n\nDone!")
    
    
