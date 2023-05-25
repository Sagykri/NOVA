
import logging
import os
import sys
import cv2
import numpy as np

sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from src.common.lib.preprocessing_utils import crop_to_tiles, segment
from cellpose import models
import cellpose
from shapely.geometry import Polygon

from src.common.lib.utils import LogDF, init_logging, xy_to_tuple
from skimage import io

nucleus_diameter = 60
tile_width = 256
tile_height = 256
cellprob_threshold = 0
flow_threshold = 0.4
min_edge_distance = 2

cp_model = models.Cellpose(gpu=True, model_type='nuclei')
logs_folder = os.path.join(os.getenv("MOMAPS_HOME"), "sandbox", "logs")


def compare():
    DAPI_path = "/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk/batch8/WT/panelA/Untreated/rep1/DAPI/"
    images_paths = [os.path.join(DAPI_path, f) for f in os.listdir(DAPI_path)]
    
    logging.info(f"Loading images...")
    images = np.asarray([(img_path, io.imread(img_path)) for img_path in images_paths])
    
    logging.info(f"images shape: {images.shape}")
    
    logging.info(f"Init LogDF")
    log = LogDF(logs_folder, columns=[
        "path", "cells_whole", "cells_tile", "cells_whole_mean", "cells_tile_mean",
        "cells_whole_std", "cells_tile_std", "valid_whole", "valid_tile" 
    ])
    
    logging.info(f"Looping images")
    for path, img in images:
        logging.info(path)
        logging.info("Whole..")
        n_cells_per_tile_whole, n_valid_tiles_whole = segment_whole_image(img)
        logging.info("Per tile..")
        n_cells_per_tile_tile, n_valid_tiles_tile = segment_per_tile(img)
        
        logging.info("Writing to log..")
        log.write([path, n_cells_per_tile_whole, n_cells_per_tile_tile,
                   round(np.mean(n_cells_per_tile_whole), 2), round(np.mean(n_cells_per_tile_tile), 2),
                   round(np.std(n_cells_per_tile_whole), 2), round(np.std(n_cells_per_tile_tile), 2),
                   n_valid_tiles_whole, n_valid_tiles_tile
                   ])
        logging.info("Next->")
        

def __segement_image(img):
    kernel = np.array([[-1,-1,-1], [-1,25,-1], [-1,-1,-1]])
    img_for_seg = cv2.filter2D(img, -1, kernel)
    masks, _, _, _ = segment(img=img_for_seg, channels=[1+1,0],\
                                    model=cp_model, diameter=nucleus_diameter,\
                                    cellprob_threshold=cellprob_threshold,\
                                    flow_threshold=flow_threshold,channel_axis=-1, show_plot=False)
        
    return masks

def segment_whole_image(img):
    img = np.stack([img, img], axis=-1)
    
    masks = __segement_image(img)
    
    # Crop masks
    masked_tiles = [masks[w:w+tile_width, h:h+tile_height] for w in range(0, masks.shape[0], tile_width) for h in range(0, masks.shape[1], tile_height)]
    masked_tiles = np.stack(masked_tiles, axis=-1)
    masked_tiles = np.moveaxis(masked_tiles, -1, 0)
    
    n_cells_per_tile = []
    n_valid_tiles = 0
    
    for masked_tile in masked_tiles:
        n_valid_tiles = __post_seg(n_cells_per_tile, n_valid_tiles, masked_tile)
        
    return n_cells_per_tile, n_valid_tiles
        
def segment_per_tile(img):
    n_cells_per_tile = []
    n_valid_tiles = 0
    
    tiles = crop_to_tiles(tile_width, tile_height, img)
    
    for tile in tiles:
        tile_masks = __segement_image(tile)
        n_valid_tiles = __post_seg(n_cells_per_tile, n_valid_tiles, tile_masks) 
    
    return n_cells_per_tile, n_valid_tiles

def __post_seg(n_cells_per_tile, n_valid_tiles, masks):
    outlines = cellpose.utils.outlines_list(masks)
    polys_nuclei = [Polygon(xy_to_tuple(o)) for o in outlines]
        
    n_cells_per_tile.append(len(polys_nuclei))
        
        # Build polygon of image's edges
    img_edges = Polygon([[min_edge_distance,min_edge_distance],\
                        [min_edge_distance,tile_height-min_edge_distance],\
                        [tile_width-min_edge_distance,tile_height-min_edge_distance],\
                        [tile_width-min_edge_distance,min_edge_distance]])
        
        # Is there any nuclues inside the image boundries?
    is_covered = [p.covered_by(img_edges) for p in polys_nuclei]
    is_valid = any(is_covered)

        #####################################################################
        ############# 210722: New constraint - only 1-5 nuclei per tile #####
    is_valid = is_valid and (len(polys_nuclei) >= 1 and len(polys_nuclei) <= 5)
        
    if is_valid:
        n_valid_tiles = n_valid_tiles + 1 if n_valid_tiles is not None else 1
    return n_valid_tiles


if __name__ == "__main__":
    init_logging(os.path.join(logs_folder, "log.log"))
    compare()
    logging.info("Done")
    