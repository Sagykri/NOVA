import os
import sys
from typing import List, Tuple
import logging
import matplotlib.pyplot as plt
import numpy as np
import cv2
import cellpose
from cellpose import models
from shapely.geometry import Polygon
import skimage.exposure

sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.common.utils import xy_to_tuple

def fit_image_shape(img:np.ndarray , expected_shape:Tuple[int,int]=(1024,1024))->np.ndarray :
    """Fit the given image shape to the expected shape given by cropping its frame if possible

    Args:
        img (np.ndarray ): The image to fit
        expected_shape (Tuple[int,int], optional): The expected shape for the image. Defaults to (1024,1024).

    Returns:
        np.ndarray : The fitted image
    """
    w, h = img.shape
    expected_w, expected_h = expected_shape
    
    # Shrinking image isn't supported
    assert w >= expected_w and h >= expected_h, f"Expected image shape to be at least the expected shape, but got {img.shape}<{expected_shape}"
    
    # If the size is as expected, return it
    if w == expected_w and h == expected_h:
        return img
    
    # Otherwise
    logging.warning(f"The image shape {img.shape} isn't as expected {expected_shape}.\nCropping image to fit")
    
    # Calculate the number of additional pixels in width and height
    w_diff:int = w - expected_w
    h_diff:int = h - expected_h
    
    # Check the number of additional pixels can be spread evenly between the image's frame to be removed
    assert (w_diff % 2 == 0) and (h_diff % 2 == 0), f"Expected count of additional pixels to be even, but got ({w_diff}, {h_diff})"
    
    # How many pixels to remove from each side in order to remove the frame
    w_diff_per_side:int = w_diff // 2
    h_diff_per_side:int = h_diff // 2
    logging.warning(f"Removing pixels from the frame ({w_diff_per_side}, {h_diff_per_side})")
    
    img = img[w_diff_per_side:-w_diff_per_side, h_diff_per_side:-h_diff_per_side]

    logging.info(f"New image shape: {img.shape}")
    return img

def rescale_intensity(img:np.ndarray , lower_bound:float=0.5, upper_bound:float=99.9)->np.ndarray :
    """Return image after stretching or shrinking its intensity levels, by setting pixels below lower_bound% to 0.0 and above upper_bound% to 1.0.\n
    For more details see: https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.rescale_intensity
    
    Args:
        img (np.ndarray ): The image to scale its intensity
        lower_bound (float, optional): The lower bound as percentile. Defaults to 0.5.
        upper_bound (float, optional): The upper bound as percentile. Defaults to 99.9.
    Returns:
        img_scaled (np.ndarray ): The scaled image with pixels intensity between 0.0 to 1.0.
    """
    
    assert 0<=lower_bound<=upper_bound<=100, "lower_bound and upper_bound must be between 0 to 100 (inclusive). upper_bound must be at least lower_bound"
    
    # Calculating the pixel instensity for the lower_bound and upper_bound percentiles in the image
    vmin, vmax = np.percentile(img, q=(lower_bound, upper_bound))
    # Map all pixels with intensity below vmin to 0.0, and all pixels with intensity above vmax to 1.0
    img_scaled = skimage.exposure.rescale_intensity(
                                                    img,
                                                    in_range=(vmin, vmax),
                                                    out_range=(0.0,1.0)
        )
    return img_scaled

def is_image_focused(img:np.ndarray , thresholds:Tuple[float, float]):
    lower_bound, upper_bound = thresholds
    assert upper_bound >= lower_bound, f"Lower bound {lower_bound} is larger than the upper bound {upper_bound}"
    
    img_focus_quality = get_image_focus_quality(img)
    
    if lower_bound <= img_focus_quality <= upper_bound:
        return True
    
    logging.warning(f"Image is blurred. Expected to be in range of ({lower_bound}, {upper_bound}), but got {img_focus_quality}")
    return False

def get_nuclei_segmentations(
    img: np.ndarray, 
    cellpose_model:models.Cellpose=None,
    diameter: float = 60.0, 
    cellprob_threshold: float = 0.0, 
    flow_threshold: float = 0.4, 
    show_plot: bool = True
) -> np.ndarray:
    """
    Segment the nucleus in an image using cellpose (https://www.cellpose.org/)
    
    Args:
        img (np.ndarray): The input image to segment.
        cellpose_model (cellpose.model.Cellpose, optional): A pre-initialized cellpose model to use. Init a new one if set to None. Defaults to None.
        diameter (float, optional): The estimated diameter of the cells to segment. Defaults to 60.0.
        cellprob_threshold (float, optional): The threshold for cell probability. Defaults to 0.0.
        flow_threshold (float, optional): The threshold for flow error. Defaults to 0.4.
        show_plot (bool, optional): Whether to display a plot of the segmentation. Defaults to True.
    
    Returns:
        masks (np.ndarray): The mask of segmented nuclei.
    """

    # Sharpen the image for easier segmentation
    sharpening_filter = np.array([[-1,-1,-1], [-1,25,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, sharpening_filter)
    
    # Segment the image using the model
    model = cellpose_model if cellpose_model is not None else models.Cellpose(gpu=True, model_type='nuclei')
    masks, flows, _, _ = model.eval(
        img, 
        diameter=diameter, 
        cellprob_threshold=cellprob_threshold, 
        flow_threshold=flow_threshold
    )
    
    # If requested, plot the segmentation results
    if show_plot:
        fig = plt.figure(figsize=(12, 5))
        
        # Display segmentation results
        cellpose.plot.show_segmentation(fig, img, masks, flows[0])
        plt.tight_layout()
        plt.show()

    return masks


def crop_image_to_tiles(
    img: np.ndarray ,
    tile_shape: Tuple[int, int]
) -> np.ndarray :
    """
    Crop the input image into tiles of the specified size (tile_shape).
    
    Args:
        img (np.ndarray): The input image to be cropped (channels, width, height).
        tile_shape (Tuple[int,int]): The shape for the tile. (width, height)
    
    Returns:
        np.ndarray: A stack of cropped image tiles (N, tile_width, tile_height, num_channels).
    """
    
    """Crop tiles to given size"""
    
    if len(img.shape) < 3:
        # Adding a channel axis if not exists
        img = np.expand_dims(img, axis=-1)
    
    # Extract number of channels
    num_channels:int = img.shape[-1]
    
    # Extract image & tile dimensions
    img_width, img_height = img.shape[:2]
    tile_width, tile_height = tile_shape
    width_residuals, height_residuals = img_width % tile_width, img_height % tile_height
    
    ########################
    ## Handle residuals ###
    ########################
    # Check if the image dimensions are divisible by tile size
    if width_residuals != 0 or height_residuals != 0:
        logging.warning(f"Image dimensions ({img_width}, {img_height}) are not divisible by tile size ({tile_width}, {tile_height}). \nCropping will result in partial tiles being excluded.")
        # Crop the image to make it divisible by the tile size
        if width_residuals != 0:
            img = img[:-width_residuals]
        if height_residuals != 0:
            img = img[:, :-height_residuals]
        
    img_width, img_height = img.shape[:2]    
    #######################
    
    # Calculate the expected number of tiles
    n_rows, n_cols = (img_width // tile_width), (img_height // tile_height)
    
    # Reshape the image into tiles
    tiles = img.reshape(
        n_rows, tile_width,
        n_cols, tile_height,
        num_channels
    ).transpose(0, 2, 1, 3, 4).reshape(-1, tile_width, tile_height, num_channels)

    # Validate the number of tiles
    n_tiles_expected = n_rows * n_cols
    assert tiles.shape[0] == n_tiles_expected, ValueError(f"Expected {n_tiles_expected} tiles but got {len(tiles)}.")

    return tiles

def extract_polygons_from_mask(mask:np.ndarray )->List[Polygon]:
    """Get polygons objects of the objects within a given mask

    Args:
        mask (np.ndarray ): The mask

    Returns:
        List[Polygon]: The list of object within the given mask as polygons
    """
    polygons = [Polygon(xy_to_tuple(o)) for o in cellpose.utils.outlines_list(mask)]
    
    return polygons
def is_contains_whole_nucleus(
    nuclei_polygons: List[Polygon], 
    tile_shape: Tuple[int, int], 
    min_edge_distance: int = 2
) -> bool:
    """
    Check if there exists a whole nucleus within the given mask that is completely inside
    the boundaries of the image, accounting for a minimum required distance from the edges.
    
    Args:
        nuclei_polygons (List[Polygon]): List of nuclei polygons.
        tile_shape (Tuple[int, int]): The shape of the tile (width, height).
        min_edge_distance (int, optional): Minimum required distance (in pixels) from the image edges. Defaults to 2.
    
    Returns:
        bool: True if a whole nucleus exists within the image boundaries, False otherwise.
    """
    is_exists = get_whole_nuclei_count(nuclei_polygons=nuclei_polygons, tile_shape=tile_shape, min_edge_distance=min_edge_distance) > 0

    return is_exists

def get_whole_nuclei_count(
    nuclei_polygons: List[Polygon] = None, 
    tile_shape: Tuple[int, int] = None, 
    min_edge_distance: int = 2,
    masked_tile: np.ndarray = None
) -> int:
    """ Count the number of whole nucleus within the given mask that is completely inside
    the boundaries of the image, accounting for a minimum required distance from the edges.    
    
    Args:
        nuclei_polygons (List[Polygon], optiona): List of nuclei polygons. 
        tile_shape (Tuple[int, int], optiona): The shape of the tile (width, height).
        min_edge_distance (int, optional): Minimum required distance (in pixels) from the image edges. Defaults to 2.
        masked_tile (np.ndarray, optional): Segmented tile for nuclei within
        
        *Note*: You may set the masked_tile or nuclei_polygons, but not both. 
        Priority is given to masked_tile.

    Returns:
        int: The number of whole nuclei in a tile size
    """
    
    if nuclei_polygons is not None and masked_tile is not None:
        logging.warning("Both 'nuclei_polygons' and 'masked_tile' are given, using 'masked_tile'")
    
    if masked_tile is not None:
        nuclei_polygons = extract_polygons_from_mask(masked_tile)
        tile_shape = masked_tile.shape[:2]
    
    tile_width, tile_height = tile_shape

    # Define the image boundary polygon
    boundries_polygon = Polygon([
        (min_edge_distance, min_edge_distance),
        (min_edge_distance, tile_height - min_edge_distance),
        (tile_width - min_edge_distance, tile_height - min_edge_distance),
        (tile_width - min_edge_distance, min_edge_distance)
    ])

    # Check if any nucleus is fully within the image boundaries
    whole_nuclei_counts = sum([p.covered_by(boundries_polygon) for p in nuclei_polygons])
    return whole_nuclei_counts

def get_nuclei_count(masked_tile: np.ndarray) -> int:
    """
    Count how many nuclei are in a tile 

    Args:
        masked_tile (np.ndarray): Segmented tile for nuclei within
    
    Returns:
        int: The number of nuclei in a tile
    """
    return len(np.unique(masked_tile))-1

def get_image_focus_quality(image: np.ndarray) -> float:
    """
    Compute the overall focus quality of an image by summing the Brenner focus
    metric along both rows and columns.

    Args:
        image (np.ndarray): 2D array representing the grayscale image.

    Returns:
        float: The overall focus score, which is the sum of Brenner focus metrics
               computed for both rows and columns.
    """
    # Calculate Brenner focus along the rows (horizontal sharpness)
    rows_brenner = __calculate_brenner_focus(image)
    # Calculate Brenner focus along the columns (vertical sharpness) by transposing the image
    cols_brenner = __calculate_brenner_focus(image.T)
    
    # Return the sum of the focus metrics for rows and columns
    return rows_brenner + cols_brenner
  
def __calculate_brenner_focus(image: np.ndarray) -> float:
    """
    Calculate the Brenner focus metric for an image by computing the squared
    differences of pixel intensities between pixels 2 positions apart.
    
    For more information see: https://patents.google.com/patent/US8014583B2/en

    Args:
        image (np.ndarray): 2D array representing the grayscale image.

    Returns:
        float: The Brenner focus score, which is the sum of squared differences
               of pixel intensities across the image.
    """
    # Typical distance between pixels for the Brenner focus metric calculation
    shift = 2  
    # Compute the difference between pixels that are 2 positions apart along the rows
    diff = image[:, :-shift] - image[:, shift:]
    # Square the differences and sum them up to get the Brenner focus score
    brenner = np.sum(diff ** 2)
    
    return brenner
