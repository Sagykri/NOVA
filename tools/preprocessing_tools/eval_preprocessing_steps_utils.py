import os
import sys
import numpy as np
import cv2 
import pandas as pd
import matplotlib.pyplot as plt 

sys.path.insert(1, os.getenv('NOVA_HOME'))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")



from src.preprocessing.preprocessing_utils import get_image_focus_quality, rescale_intensity, fit_image_shape, get_nuclei_segmentations
from tools.preprocessing_tools.image_sampling_utils import sample_raw_images, sample_processed_images

def plot_images(images, paths, n_samples=3, expected_site_width=1024, expected_site_height=1024, figsize=(16,8), suptitle=None):
    fig, ax = plt.subplots(1, n_samples, figsize=figsize)
    if suptitle is not None:
        fig.suptitle(suptitle)
    
    def __plot(ax, image, path):
        ax.set_title(f'{os.path.basename(path)}')
        put_tiles_grid(ax, expected_site_width, expected_site_height)
        ax.imshow(image, cmap='gray')
        ax.set_axis_off()
    
    if len(images) == 1:
        __plot(ax, images[0], paths[0])
    else:    
        for i in range(n_samples):
            __plot(ax[i], images[i], paths[i])
        
    plt.show()
    
def plot_processed_images(images, paths, n_samples=3, figsize=(16,8)):
    for i in range(n_samples):
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].set_title(f'{os.path.basename(paths[i])} - Target', fontsize=11)
        ax[0].imshow(images[i,...,0], cmap='gray', vmin=0, vmax=1)
        ax[0].set_axis_off()
        
        ax[1].set_title(f'{os.path.basename(paths[i])} - Nucleus', fontsize=11)
        ax[1].imshow(images[i,...,1], cmap='gray', vmin=0, vmax=1)
        ax[1].set_axis_off()
        plt.show()
    
def put_tiles_grid(ax, w, h, block_size=128):
    # Add dashed grid lines
    num_blocks = w / block_size
    for i in range(1, num_blocks):
        # Draw horizontal dashed lines
        ax.plot([0, w], [i * block_size, i * block_size], linestyle='--', lw=1, alpha=0.5, color='pink')

        # Draw vertical dashed lines
        ax.plot([i * block_size, i * block_size], [0, h], linestyle='--', lw=1, alpha=0.5, color='pink')

def test_cellpose(images, images_paths, flow_threshold=0.4, cellprob_threshold=0, diameter=60):
    from cellpose import models
    import cellpose
    from shapely.geometry import Polygon
    
    cp_model = models.Cellpose(gpu=True, model_type='nuclei')
    for i, image in enumerate(images):
        img = np.stack([image, image], axis=-1)
        masks = get_nuclei_segmentations(img=img, cellpose_model=cp_model,show_plot=True, flow_threshold=flow_threshold,
                                         cellprob_threshold=cellprob_threshold, diameter=diameter) #channel_axis=-1,
        # flow_threshold - The default is flow_threshold=0.4. Increase this threshold if cellpose is not returning as many ROIs as you’d expect. 
        #                   Similarly, decrease this threshold if cellpose is returning too many ill-shaped ROIs.
        # cellprob_threshold - The default is cellprob_threshold=0.0. Decrease this threshold if cellpose is not returning as many ROIs as you’d expect. 
        #                       Similarly, increase this threshold if cellpose is returning too ROIs particularly from dim areas.
        binary_mask = masks.copy()
        binary_mask[binary_mask>0] = 1
        fig, axs = plt.subplots(ncols=2)
        axs[0].imshow(image,cmap='gray')#, vmin=0, vmax=1)
        axs[0].axis('off')
        axs[1].imshow(masks)#, cmap='gray')
        axs[1].axis('off')
        axs[1].set_title(f'Segmented {len(np.unique(masks))-1} objects')
        plt.tight_layout()
        plt.show()

def check_preprocessing_steps(input_dir_batch, sample_size, brenner_path, marker, cell_line, condition, rep, panel=None,
                              expected_site_width=1024,expected_site_height=1024):
    
    # Sample images for marker
    # Get paths
    images_paths = sample_raw_images(input_dir_batch, marker,
                                 cell_line=cell_line, condition=condition, sample_size=sample_size, rep=rep, panel=panel)
    

    # Load images
    images = []
    for p in images_paths:
        image = cv2.imread(p, cv2.IMREAD_ANYDEPTH)
        images.append(image )
        
    images = np.vstack(images)
        
    images_processed = np.zeros((images.shape[0], expected_site_width, expected_site_height))
    images_paths_processed = images_paths.copy()
    
    plot_images(images_processed, images_paths_processed)
    
    # Handle image sizes
    
    for i in range(len(images)):
        images_processed[i] = fit_image_shape(images[i], (expected_site_width, expected_site_height))
    
    plot_images(images_processed, images_paths_processed)
    
    # Rescale intensity and plot
    for i in range(len(images_processed)):
        images_processed[i] = rescale_intensity(images_processed[i])
        
    plot_images(images_processed, images_paths_processed)

    # Plot images which passes and failed the brenner
    brenner_cutoffs = pd.read_csv(brenner_path, index_col=0)

    brenner_marker = brenner_cutoffs.loc[marker]
    brenner_lower_bound, brenner_upper_bound = brenner_marker[0], brenner_marker[1]
        
    brenner_for_sampled_images = []

    for i in range(len(images_processed)):
        brenner_for_sampled_images.append(get_image_focus_quality(images_processed[i]))
    brenner_for_sampled_images = np.asarray(brenner_for_sampled_images)

    # Show images passed filter
    passed_images_indexes = np.argwhere((brenner_for_sampled_images >= brenner_lower_bound) & (brenner_for_sampled_images <= brenner_upper_bound)).reshape(-1)
    if len(passed_images_indexes) > 0:
        plot_images(images_processed[passed_images_indexes], images_paths_processed[passed_images_indexes], n_samples=min(3, len(passed_images_indexes)), suptitle='Valid')
    else:
        print("No valid files (in terms of Brenner)")

    # Show images failed filter
    failed_images_indexes = list(set(np.arange(len(images_processed))) - set(passed_images_indexes))
    if len(failed_images_indexes) > 0:
        plot_images(images_processed[failed_images_indexes], images_paths_processed[failed_images_indexes], n_samples=min(3, len(failed_images_indexes)), suptitle='Invalid')
    else:
        print("All files are valid (in terms of Brenner)")

    images_processed = images_processed[passed_images_indexes]
    images_paths_processed = images_paths_processed[passed_images_indexes]


    # Plot cellpose result
    test_cellpose(images_processed, images_paths_processed)

def get_processed_images(input_dir_batch, sample_size, marker, cell_line, condition, rep, figsize=(20,8), plot=True, shuffle=True):
    processes_images_sites_paths = sample_processed_images(input_dir_batch, marker, cell_line, condition, sample_size, rep=rep)

    images = []
    processes_images_tiles_paths = []

    for p in processes_images_sites_paths:
        image_site = np.load(p)
        images.append(image_site)
        processes_images_tiles_paths.append([f'{p}_{i}' for i in range(len(image_site))])
    images = np.vstack(images)
    processes_images_tiles_paths = np.hstack(processes_images_tiles_paths)
    
    if shuffle:
        indx = np.arange(len(images))
        np.random.shuffle(indx)
        images = images[indx]
        processes_images_tiles_paths = processes_images_tiles_paths[indx]

    images = images[:sample_size]
    processes_images_tiles_paths = processes_images_tiles_paths[:sample_size]

    if plot:
        plot_processed_images(images, processes_images_tiles_paths, n_samples=sample_size, figsize=figsize)
    
    return images, processes_images_tiles_paths
    
