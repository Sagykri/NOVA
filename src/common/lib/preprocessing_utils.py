import datetime
import glob
import os
import sys
import timeit

import pandas as pd

sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import logging
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from skimage.measure import block_reduce
from src.common.lib.utils import xy_to_tuple
from src.common.lib.image_metrics import calculate_image_sharpness_brenner, calculate_var
import cv2
import cellpose
from cellpose import models
from shapely.geometry import Polygon
from skimage import io
import skimage.exposure
from pathlib import Path

def crop_frame(original_image, w=12,h=12):

    # Crop the image by removing a 12-pixel frame from each side

    cropped_image = original_image[w:-w, h:-h] 

    return cropped_image

def filter_invalid_tiles(file_name, img, nucleus_diameter=100, cellprob_threshold=0,\
                          flow_threshold=0.7, cell_inclusion_prct = 0.85, tile_w=100,tile_h=100,
                          calculate_nucleus_distance=False,
                          cp_model=None, show_plot=True, return_counts=False):
    """
    Filter invalid tiles (leave only tiles with #nuclues (not touching the edges) == 1)
    This function also segment the cells on the whole image and returns the indexes for the valid tiles
    """
    
    tiles_passed_indexes = []
    n_cells_per_tile, n_whole_cells_per_tile = [], []

    # From 1/None channels to 2 for cellpose (with some adjustments cellpose might work with single channel as well)
    img = np.stack([img, img], axis=-1)
    
    logging.info(f"[{file_name}] Segmenting nuclues, calculate_nucleus_distance {calculate_nucleus_distance}")
    
    kernel = np.array([[-1,-1,-1], [-1,25,-1], [-1,-1,-1]])
    img_for_seg = cv2.filter2D(img, -1, kernel)
    cp_model = models.Cellpose(gpu=True, model_type='nuclei') if cp_model is None else cp_model
    masks, _, _, _ = segment(img=img_for_seg, channels=[1+1,0],\
                                    model=cp_model, diameter=nucleus_diameter,\
                                    cellprob_threshold=cellprob_threshold,\
                                    flow_threshold=flow_threshold,channel_axis=-1, show_plot=show_plot)
    
    if calculate_nucleus_distance:
        # calc nuclear distance
        masks[masks!=0] = 1 # convert mask to binary
        nucleus_distance = cv2.distanceTransform(masks.astype('uint8'), cv2.DIST_L2,0)
    else:
        nucleus_distance = None

    # Calculate cells per site
    n_cells_per_site = np.max(masks)

    # Crop masks
    masked_tiles = [masks[w:w+tile_w, h:h+tile_h] for w in range(0, masks.shape[0], tile_w) for h in range(0, masks.shape[1], tile_h)]
    masked_tiles = np.stack(masked_tiles, axis=-1)
    masked_tiles = np.moveaxis(masked_tiles, -1, 0)
    
    n_masked_tiles = masked_tiles.shape[0]
    
    if show_plot:
        tiles = crop_to_tiles(tile_w, tile_h, img)
    
    hist, bins = np.histogram(masks, bins=range(masks.max()+2))
    value_counts_dict = dict(zip(bins[:-1], hist))
    
    for i in range(n_masked_tiles):
        mask = masked_tiles[i]

        # Nuclues seg
        logging.info(f"[{file_name}] Tile number {i} out of {n_masked_tiles}")
        
        if show_plot:
            _, ax = plt.subplots(1,2)
            ax[0].imshow(tiles[i,...,0], vmin=0, vmax=1)
            ax[1].imshow(tiles[i,...,1], vmin=0, vmax=1)
            plt.show()

        """
        Filter tiles with no nuclues
        """
        is_valid = False
        cur_n_whole_cells = 0
        current_nucs = np.unique(mask)
        if current_nucs.size == 1: # no object found in this tile but background (by cellpose)
            n_cells_per_tile.append(0)
            n_whole_cells_per_tile.append(0)
            continue

        for current_nuc in current_nucs:
            if current_nuc == 0 :
                continue

            current_nuc_count = np.count_nonzero(mask == current_nuc)
            if (current_nuc_count / value_counts_dict[current_nuc]) >= cell_inclusion_prct:
                is_valid = True
                cur_n_whole_cells+=1
        
        outlines = cellpose.utils.outlines_list(mask)
        polys_nuclei = [Polygon(xy_to_tuple(o)) for o in outlines]
        
        #####################################################################
        ############# 210722: New constraint - only 1-5 nuclei per tile #####
        is_valid = is_valid and (len(polys_nuclei) >= 1 and len(polys_nuclei) <= 5)
        #####################################################################

        n_cells_per_tile.append(len(current_nucs)-1)
        n_whole_cells_per_tile.append(cur_n_whole_cells)

        if is_valid:
            tiles_passed_indexes.append(i)
        
    n_cells_per_tile = np.asarray(n_cells_per_tile)
    n_whole_cells_per_tile = np.asarray(n_whole_cells_per_tile)
    tiles_passed_indexes = np.asarray(tiles_passed_indexes)

    if len(tiles_passed_indexes) == 0:
        logging.warning(f"Nothing is valid (total: {n_masked_tiles})")
        
    logging.info(f"#ALL {n_masked_tiles}, #Passed {tiles_passed_indexes.shape[0]}")
    
    if return_counts:
        return tiles_passed_indexes, n_cells_per_tile, n_whole_cells_per_tile, nucleus_distance, n_cells_per_site
    
    return tiles_passed_indexes

def rescale(n_channels, image_processed_tiles_passed, size):
    """Rescale images to given size"""
    
    image_processed_tiles_passed_reshaped = []
    for img in image_processed_tiles_passed:
        imgs = None
        for c in range(n_channels):
            img_c = np.array(img[...,c], dtype=np.float64)
            # Downsampling
            img_c = block_reduce(image=img_c, block_size=img_c.shape[0] // size, func=np.mean)

            if imgs is None:
                imgs = img_c
            else:
                imgs = np.dstack((imgs, img_c))

        image_processed_tiles_passed_reshaped.append(imgs)
            
    image_processed_tiles_passed_reshaped = np.stack(image_processed_tiles_passed_reshaped, axis=-1)
    image_processed_tiles_passed_reshaped = np.moveaxis(image_processed_tiles_passed_reshaped, -1, 0)

    return image_processed_tiles_passed_reshaped

def crop_to_tiles(tile_w, tile_h, img_processed):
    
    """Crop tiles to given size"""
    
    image_dim = len(img_processed.shape)
    if image_dim == 2:
        # If img has no channel axis, fake one
        img_processed = np.stack([img_processed, img_processed], axis=image_dim)
        image_dim = len(img_processed.shape)
        
    image_processed_tiles = []
    image_w = img_processed.shape[0]
    image_h = img_processed.shape[1]
    
    to_validate = True
    if image_w % tile_w != 0 or image_h % tile_h != 0:
        to_validate = False
        logging.warning(f"[Warning] Fuzzy divizion ({tile_w}, {tile_h}; {image_w}, {image_h})")

    img_processed = img_processed[:image_w - image_w % tile_w, :image_h - image_h % tile_h]
    image_w = img_processed.shape[0]
    image_h = img_processed.shape[1]

    if to_validate:
      n_tiles_expected = (image_w * image_w) // (tile_w * tile_h)

    for w in range(0, image_w, tile_w):
      for h in range(0, image_h, tile_h):
        image_processed_tiles.append(img_processed[w:w+tile_w, h:h+tile_h, :])

    image_processed_tiles = np.stack(image_processed_tiles, axis=image_dim)
    image_processed_tiles = np.moveaxis(image_processed_tiles, -1, 0)

    if to_validate and n_tiles_expected != image_processed_tiles.shape[0]:
      raise f"Error: #Expected tiles ({n_tiles_expected}) != #Observer tiles ({image_processed_tiles.shape[0]})"

    return image_processed_tiles

def rescale_intensity(img_current_channel):
    """Return image after stretching or shrinking its intensity levels.

    The desired intensity range of the input and output, in_range and out_range respectively, 
    are used to stretch or shrink the intensity range of the input image
    
    see: https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.rescale_intensity
    
    Args:
        img_current_channel (numpy ndarray): the image to scale

    Returns:
        img_scaled (numpy ndarray): image in scale of [0,1] after rescaling
    """
    
    vmin, vmax = np.percentile(img_current_channel, q=(0.5, 99.9))
    img_scaled = skimage.exposure.rescale_intensity(
                                                    img_current_channel,
                                                    in_range=(vmin, vmax),
                                                    out_range=np.float32
        )
    return img_scaled

def normalize(show, img_current_channel):
    """Normalize (min-max) given image"""
    
    image_test_min = np.min(img_current_channel)
    image_test_max = np.max(img_current_channel)
    img_current_channel -= image_test_min
    img_current_channel /= (image_test_max - image_test_min)
    if show:  
      plt.imshow(img_current_channel)
      plt.show()
      
    return img_current_channel

def downsample(show, img_current_channel, block_size=2):
    """Downsampling given image by 2"""
    img_current_channel = block_reduce(image=img_current_channel, block_size=block_size, func=np.mean)
    if show:
      plt.imshow(img_current_channel)
      plt.show()
    return img_current_channel

def denoise(show, img):
    """Denoising given image"""
    
    img = cv2.fastNlMeansDenoising(img,None,3,7,21)
    logging.info(img.shape)
    if show:
      _, ax = plt.subplots(1,2)
      ax[0].imshow(img[...,0])
      ax[1].imshow(img[...,1])
      plt.show()
    return img

def segment(img, model, channels=None, 
            diameter=500, cellprob_threshold=0,\
            flow_threshold=0.7,\
            channel_axis=-1, show_plot=True):
  """Segment the nucleus"""
  
  logging.info(f"Image shape: {img.shape}")
  logging.info(f"Switching axis {channel_axis} with -1:")

  img = np.moveaxis(img, channel_axis, -1)
  
  logging.info(f"Image shape (after switching axis): {img.shape}")

  """
  channels:
  First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
            Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
  -----

  Usage of channels:
  if channels[0]==0:
    ....
  else:
    chanid = [channels[0]-1]
    if channels[1] > 0:
        chanid.append(channels[1]-1)
    data = data[...,chanid]
  https://github.com/MouseLand/cellpose/blob/196278e12c135135aff3535d7d0e9218dd2eefc2/cellpose/transforms.py#L341
  """
  masks, flows, styles, diams = model.eval(img, diameter=diameter, channels=channels, cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold)
  if show_plot:
    fig = plt.figure(figsize=(12,5))
    cellpose.plot.show_segmentation(fig, img[...,channels[0]-1], masks, flows[0], channels=channels)
    plt.tight_layout()
    plt.show()

  return masks, flows, styles, diams

def __is_site_brenner_valid(img, marker_name, site_brenner_bounds):
    logging.info(f"marker_name: {marker_name}")
    if marker_name not in site_brenner_bounds.index:
        logging.info(f"Marker couldn't be found in the brenner bounds file. Passing it..")
        return True
    
    img_brenner = calculate_image_sharpness_brenner(img)
    marker_brenner_bounds = site_brenner_bounds.loc[marker_name]
    lower_bound = marker_brenner_bounds['Site_brenner_lower_bound']
    upper_bound = marker_brenner_bounds['Site_brenner_upper_bound']
    logging.info(f"Site's Brenner's gradient value: {img_brenner}; bounds for marker {marker_name}: ({lower_bound},{upper_bound})")
    if img_brenner < lower_bound or img_brenner > upper_bound:
        logging.info(f"Outlier site detected, Brenner's gradient {img_brenner} is outside of bounds ({lower_bound},{upper_bound}) for marker {marker_name}")
        return False
    
    logging.info(f"The site (for marker {marker_name}) has passed the Brenner test")
    
    return True
    
def preprocess_image_pipeline(img, save_path, n_channels=2,
                                  tiles_indexes=None,
                                  tile_width=100, tile_height=100, to_downsample=True,
                                  to_denoise=False, to_normalize=True, to_show=False, brenner_bounds=None):
    """
        Run the image preprocessing pipeline for spinning disk (spd) images
    """
    if to_denoise:
        img = denoise(to_show, True, img)   

    img_processed = None
    for c in range(n_channels):
        img_current_channel = np.array(img[...,c], dtype=np.float64)
                     
        if to_normalize:
            # Normalize original raw image
            img_current_channel = rescale_intensity(img_current_channel)
            
        if img_processed is None:
            img_processed = img_current_channel
        else:
            img_processed = np.dstack((img_processed, img_current_channel))
        
    logging.info(f"Image (post image processing) shape {img_processed.shape}")


    ############################
    # SAGY 041223
    # Filter bad sites using Brenner gradient
    marker_name = save_path.split(os.sep)[-2]
    if brenner_bounds is not None and not __is_site_brenner_valid(img_processed[...,0], marker_name, brenner_bounds):
        return []
    
    ############################
    
    # Crop 1024x1024 image to 16 tiles, each of 256x256
    image_processed_tiles = crop_to_tiles(tile_width, tile_height, img_processed)
    
    if tiles_indexes is not None:
        # Filter tiles
        image_processed_tiles = image_processed_tiles[tiles_indexes,...]
    
    image_downsampled_tiles = []
    for image_tile in image_processed_tiles:
        tile_processed = None
        for c in range(n_channels):
            tile_current_channel = np.array(image_tile[...,c], dtype=np.float64)
            
            if to_downsample:
                # Downsampling
                tile_current_channel = downsample(to_show, tile_current_channel, block_size=2) 
                
            if tile_current_channel.shape[1] != 100:
                # Resize from 128x128 to 100x100
                tile_current_channel = transform.resize(tile_current_channel, (100, 100), anti_aliasing=True)
            
            # Min max scaling
            tile_min = np.min(tile_current_channel, axis=None)
            tile_current_channel = (tile_current_channel - tile_min) / (np.max(tile_current_channel, axis=None) - tile_min)
            logging.info(f"Tile (c) min, max: {np.min(tile_current_channel)}, {np.max(tile_current_channel)}")
                        
            if tile_processed is None:
                tile_processed = tile_current_channel
            else:
                tile_processed = np.dstack((tile_processed, tile_current_channel))
                image_downsampled_tiles.append(tile_processed)
            
    # Save processed tiles to file
    with open(f"{save_path}_processed.npy", 'wb') as f:
        np.save(f, image_downsampled_tiles)
        logging.info(f"Saved to {save_path}_processed.npy")

    return image_downsampled_tiles

def preprocess_panel(slf, panel, input_folder_root,
                     output_folder_root, input_folder_root_cell_line, cp_model, raw_f, cell_line,
                     logging_df, timing_df):
        
    logging.info(f"[{raw_f} {cell_line} {panel}] Panel: {panel}")
                
    input_folder_root_panel = os.path.join(input_folder_root_cell_line, panel)
    
    conditions = [f for f in os.listdir(input_folder_root_panel) 
                if os.path.isdir(os.path.join(input_folder_root_panel, f)) and f != 'experiment setup']   
        
    logging.info(f"[{raw_f} {cell_line} {panel}] Conditions: {conditions}")
    
    for condition in conditions:    
        logging.info(f"[{raw_f} {cell_line} {panel} {condition}] Condition: {condition}")
            
        input_folder_root_condition = os.path.join(input_folder_root_panel, condition)
                        
        reps = [f for f in os.listdir(input_folder_root_condition ) if os.path.isdir(os.path.join(input_folder_root_condition , f))]

        input_folders = [os.path.join(input_folder_root, cell_line, panel, condition, rep) for rep in reps]     
        output_folders = [os.path.join(output_folder_root, cell_line, condition) for rep in reps]
        
        logging.info(f"Input folders: {input_folders}")

        format_output_filename = lambda filename, ext: f"{filename}_{panel}_{cell_line}{ext}"
        
        for input_folder, output_folder in zip(input_folders, output_folders):
            markers = os.listdir(input_folder)
            rep = os.path.basename(input_folder)
            nucleus_folder = os.path.join(input_folder, "DAPI")
            valid_tiles_indexes = {}
            
            start_time = timeit.default_timer()
            
            for marker in markers:
                if slf.markers_to_include is not None and marker not in slf.markers_to_include:
                    logging.info(f"Skipping {marker}")
                    continue
                        
                input_subfolder = os.path.join(input_folder, marker)
                output_subfolder = os.path.join(output_folder, marker)
                
                logging.info(f"Marker: {marker}")
                logging.info(f"Subfolder {input_subfolder}")
                
                
                for f in os.listdir(input_subfolder):
                    filename, ext = os.path.splitext(f)
                    output_filename = format_output_filename(filename, '') if format_output_filename else f
                    save_path = os.path.join(output_subfolder, f"{rep}_{output_filename}")
                    
                    # skip the "tiles validation" if file in this name already exist. 
                    logging.info(f"Save path: {save_path}")
                    if os.path.exists(f"{save_path}_processed.npy"): 
                        logging.info(f"[Skipping ,exists] Already exists {save_path}_processed")
                        continue
                    
                    if slf.conf.SELECTIVE_INPUT_PATHS is not None \
                        and os.path.join(input_subfolder, f) not in slf.conf.SELECTIVE_INPUT_PATHS:
                        logging.info(f"Skipping {os.path.join(input_subfolder, f)} since not in SELECTIVE_INPUT_PATHS")
                        continue
                    
                    if ext != '.tif':
                        continue
                    
                    site = filename.split('_')[-1]
                    target_filepath = os.path.join(input_subfolder, f)
                    
                    nucleus_filepath = glob.glob(f"{nucleus_folder}/*_{site}{ext}")
                    if len(nucleus_filepath) == 0:
                        logging.info(f"Skipping site {site} for {target_filepath} since no DAPI for this site was found")
                        continue
                    
                    nucleus_filepath = nucleus_filepath[0]
                    logging.info(f"{target_filepath}, {nucleus_filepath}")
                                        
                    if site not in valid_tiles_indexes:
                        nucleus_diameter        = slf.nucleus_diameter
                        tile_width              = slf.tile_width
                        tile_height             = slf.tile_height
                        cellprob_threshold      = slf.cellprob_threshold
                        flow_threshold          = slf.flow_threshold
                        cell_inclusion_prct     = slf.cell_inclusion_prct
                        to_show                 = slf.to_show
                        with_nucelus_distance   = slf.conf.WITH_NUCLEUS_DISTANCE
                        crop_frame_size         = slf.crop_frame_size
                        brenner_bounds          = slf.brenner_bounds

                        # Crop DAPI tiles
                        img_nucleus = cv2.imread(nucleus_filepath, cv2.IMREAD_ANYDEPTH) #used to be IMREAD_GRAYSCALE
                        
                        ############################
                        # SAGY 041223
                        logging.info(f"Cropping DAPI by ({crop_frame_size[0]}, {crop_frame_size[1]}) for (w,h)")
                        img_nucleus = crop_frame(img_nucleus, crop_frame_size[0], crop_frame_size[1]) 
                        ############################
                        
                        logging.info("Rescaling intensity of DAPI")
                        img_nucleus = rescale_intensity(img_nucleus)
                        
                        # Filter bad sites
                        # If site doesn't pass Brenner's thresholds - filter it out
                        logging.info("Filtering bad site in DAPI by Brenner")
                        if brenner_bounds is None:
                            raise "brenner_bounds is None"
                        
                        if brenner_bounds is not None and not __is_site_brenner_valid(img_nucleus, 'DAPI', brenner_bounds):
                            logging.warning(f"Nothing is valid due to Brenner bounds")
                            valid_tiles_indexes[site] =  np.asarray([])
                        else:
                            # Filter invalid tiles (keep tiles with at least one full nuclues (nuclues border is not overlapping image edges))
                            logging.info("Filtering bad tiles in DAPI (cellpose)")
                            current_valid_tiles_indexes, n_cells_per_tile, n_whole_cells_per_tile, nucleus_distance, n_cells_per_site = filter_invalid_tiles(nucleus_filepath,
                                                                                                                                                            img_nucleus,
                                                                                                                                                            nucleus_diameter=nucleus_diameter, 
                                                                                                                                                            cellprob_threshold=cellprob_threshold,
                                                                                                                                                            flow_threshold=flow_threshold, 
                                                                                                                                                            cell_inclusion_prct = cell_inclusion_prct,
                                                                                                                                                            tile_w=tile_width, tile_h=tile_height, 
                                                                                                                                                            cp_model=cp_model,
                                                                                                                                                            calculate_nucleus_distance=with_nucelus_distance,
                                                                                                                                                            show_plot=to_show,
                                                                                                                                                            return_counts=True)
                            
                            logging.info(f"[{nucleus_filepath}] {len(current_valid_tiles_indexes)}")# out of {len(nucleus_image_tiles)} passed ({len(nucleus_image_tiles)-len(current_valid_tiles_indexes)} invalid)")
                            
                            # Save the indexes of the valid tiles for current site
                            valid_tiles_indexes[site] = current_valid_tiles_indexes
                                                    
                            logging.info(f"[{nucleus_filepath}] Saving DAPI stats to file {logging_df.path}")
                            to_log = [datetime.datetime.now().strftime("%d%m%y_%H%M%S"), filename, raw_f, cell_line,
                                                panel, condition, rep, "DAPI",
                                                n_cells_per_tile,
                                                
                                                current_valid_tiles_indexes, # SAGY 201123 
                                                
                                                round(np.mean(n_cells_per_tile), 2), round(np.std(n_cells_per_tile), 2),
                                                n_whole_cells_per_tile,
                                                round(np.mean(n_whole_cells_per_tile), 2), round(np.std(n_whole_cells_per_tile), 2),
                                                len(current_valid_tiles_indexes),
                                                n_cells_per_site]
                                                
                            
                            if len(current_valid_tiles_indexes) > 0:
                                to_log += [round(np.mean(n_cells_per_tile[current_valid_tiles_indexes]), 2),
                                        round(np.std(n_cells_per_tile[current_valid_tiles_indexes]), 2),
                                            round(np.mean(n_whole_cells_per_tile[current_valid_tiles_indexes]), 2),
                                            round(np.std(n_whole_cells_per_tile[current_valid_tiles_indexes]), 2)]
                            else:
                                to_log += [None]*4
                                
                            logging_df.write(to_log)
                        
                    else:
                        logging.info(f"[Marker {marker}, Site: {site}] Valid tiles have already been calculated ({valid_tiles_indexes[site]})")    
                
                    logging.info(output_subfolder)
                    if not os.path.exists(output_subfolder):
                        logging.info(f"[Creating subfolder]  {output_subfolder} {os.path.exists(output_subfolder)}")
                        Path(output_subfolder).mkdir(parents=True, exist_ok=True)

                    tiles_indexes = valid_tiles_indexes[site]
                    
                    if len(tiles_indexes) == 0:
                        logging.warning(f"No valid tiles for {os.path.join(input_subfolder, f)}")
                        continue

                    processed_images = slf.preprocess_image(target_filepath, save_path,
                                                nucleus_file=nucleus_filepath,
                                                img_nucleus=nucleus_distance,
                                                tiles_indexes=tiles_indexes,
                                                show=slf.to_show,
                                                flow_threshold=slf.flow_threshold)
                    
                    if len(processed_images) > 0 and marker != 'DAPI':
                        logging.info(f"[{nucleus_filepath}] Saving target stats to file {logging_df.path}")
                        to_log = [datetime.datetime.now().strftime("%d%m%y_%H%M%S"), filename, raw_f, cell_line,
                                            panel, condition, rep, marker,
                                            None,
                                            
                                            tiles_indexes, # SAGY 201123 
                                            
                                            None, None,
                                            None,
                                            None, None,
                                            len(tiles_indexes),
                                            None]
                                            
                        to_log += [None]*4
                            
                        logging_df.write(to_log)
                    
            elapsed_time = timeit.default_timer() - start_time
            logging.info(f"[{raw_f}, {cell_line}, {panel}, {condition}, {rep}] Saving timing to file {timing_df.path}")
            timing_df.write([datetime.datetime.now().strftime("%d%m%y_%H%M%S"), raw_f, cell_line,
                                panel, condition,
                                rep, elapsed_time
                            ])

