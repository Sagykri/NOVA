import logging
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import block_reduce
from src.common.lib.utils import xy_to_tuple
import cv2
import cellpose
from cellpose import models
from shapely.geometry import Polygon


def filter_invalid_tiles(file_name, tiles, nucleus_diameter=100, cellprob_threshold=0,\
                          flow_threshold=0.7, min_edge_distance = 2, tile_w=100,tile_h=100, show_plot=True):
    """
    Filter invalid tiles (leave only tiles with #nuclues (not touching the edges) == 1)
    """
    image_processed_tiles_passed = []
    n_tiles = tiles.shape[0]


    for i in range(n_tiles):
        tile = tiles[i]

        # Nuclues seg
        logging.info(f"[{file_name}] Tile number {i} out of {n_tiles}")
        logging.info(f"[{file_name}] Segmenting nuclues")
        
        if show_plot:
            _, ax = plt.subplots(1,2)
            ax[0].imshow(tile[...,0])
            ax[1].imshow(tile[...,1])
            plt.show()

        kernel = np.array([[-1,-1,-1], [-1,25,-1], [-1,-1,-1]])
        tile_for_seg = cv2.filter2D(tile, -1, kernel)

        seg_save_path = f'{file_name}_nuclei'
        masks, _, _, _ = segment(img=tile_for_seg, channels=[1+1,0],\
                                        model_type='nuclei', diameter=nucleus_diameter,\
                                        cellprob_threshold=cellprob_threshold,\
                                        flow_threshold=flow_threshold,save_path=seg_save_path, channel_axis=-1, show_plot=show_plot)


        """
        Filter tiles with no nuclues
        """
        outlines = cellpose.utils.outlines_list(masks)
        polys_nuclei = [Polygon(xy_to_tuple(o)) for o in outlines]

        # Build polygon of image's edges
        img_edges = Polygon([[min_edge_distance,min_edge_distance],\
                        [min_edge_distance,tile_h-min_edge_distance],\
                        [tile_w-min_edge_distance,tile_h-min_edge_distance],\
                        [tile_w-min_edge_distance,min_edge_distance]])
        
        # Is there any nuclues inside the image boundries?
        is_valid = any([p.covered_by(img_edges) for p in polys_nuclei])

        #####################################################################
        ############# 210722: New constraint - only 1-5 nuclei per tile #####
        is_valid = is_valid and (len(polys_nuclei) >= 1 and len(polys_nuclei) <= 5)
        #####################################################################


        if is_valid:
            image_processed_tiles_passed.append(tile)

    if len(image_processed_tiles_passed) == 0:
        logging.info(f"Nothing is valid (total: {n_tiles})")
        
        return np.array(image_processed_tiles_passed)
        
    image_processed_tiles_passed = np.stack(image_processed_tiles_passed, axis=-1)
    image_processed_tiles_passed = np.moveaxis(image_processed_tiles_passed, -1,0)

    logging.info(f"#ALL {n_tiles}, #Passed {image_processed_tiles_passed.shape[0]}")

    return image_processed_tiles_passed

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

def downsample(show, img_current_channel):
    """Downsampling given image by 2"""
    img_current_channel = block_reduce(image=img_current_channel, block_size=2, func=np.mean)
    logging.info(img_current_channel.shape)
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

def segment(img, channels=None, diameter=500,\
                     model_type='cyto2', cellprob_threshold=0,\
                     flow_threshold=0.7,\
                     channel_axis=-1, show_plot=True, save_path=None):
  """Segment the nucleus"""
  
  model = models.Cellpose(gpu=True, model_type=model_type)
  
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

  if save_path is not None:
    # save results
    cellpose.io.masks_flows_to_seg(img, masks, flows, diams, save_path, channels)
    
    logging.info(f"Saved to {save_path}")

  if show_plot:
    fig = plt.figure(figsize=(12,5))
    cellpose.plot.show_segmentation(fig, img[...,channels[0]-1], masks, flows[0], channels=channels)
    plt.tight_layout()
    plt.show()

  return masks, flows, styles, diams


def preprocess_image_pipeline(file_path, save_path, n_channels=2, nucleus_diameter=60,
                              flow_threshold=0.4, cellprob_threshold=0, min_edge_distance=2,
                              tile_width=100, tile_height=100, to_downsample=True,
                              to_denoise=False, to_normalize=True, to_show=False):
    """
        Run the image preprocessing pipeline
    """
    if to_denoise:
        img = denoise(to_show, True, img)   

    img_processed = None
    for c in range(n_channels):
        img_current_channel = np.array(img[...,c], dtype=np.float64)
        
        if to_downsample:
            # Downsampling
            img_current_channel = downsample(to_show, img_current_channel) 

        if to_normalize:
            # Normalize
            normalize(to_show, img_current_channel)

        if img_processed is None:
            img_processed = img_current_channel
        else:
            img_processed = np.dstack((img_processed, img_current_channel))
        
    logging.info(f"Image (post image processing) shape {img_processed.shape}")

    # Crop tiles
    image_processed_tiles = crop_to_tiles(tile_width, tile_height, img_processed)

    # Filter invalid tiles (leave only tiles with #nuclues (not touching the edges) == 1)
    image_processed_tiles_passed = filter_invalid_tiles(file_path, image_processed_tiles,\
                                                        nucleus_diameter=nucleus_diameter, cellprob_threshold=cellprob_threshold,\
                                                        flow_threshold=flow_threshold, min_edge_distance = min_edge_distance,\
                                                            tile_w=tile_width,tile_h=tile_height, show_plot=to_show)
    
    logging.info(f"[{file_path}] {len(image_processed_tiles_passed)} out of {len(image_processed_tiles)} passed ({len(image_processed_tiles)-len(image_processed_tiles_passed)} unvalid)")
    
    if len(image_processed_tiles_passed) == 0:
        logging.info(f"[{file_path}] No valid results. Skipping this one")
        return image_processed_tiles_passed
    
    size = 100
    if image_processed_tiles_passed.shape[1] != size:
        # Reshape to sizexsize (100x100)
        logging.info(f"Reshape tiles to {size}x{size} (block_size={image_processed_tiles_passed.shape[1] // size})")
        
        image_processed_tiles_passed_reshaped = rescale(n_channels, image_processed_tiles_passed, size)
    else:
        image_processed_tiles_passed_reshaped = image_processed_tiles_passed

    # Save processed tiles to file
    with open(f"{save_path}_processed", 'wb') as f:
        np.save(f, image_processed_tiles_passed_reshaped)
        logging.info(f"Saved to {save_path}_processed.npy")

    return image_processed_tiles_passed_reshaped