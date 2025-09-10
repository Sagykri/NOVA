import os
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")
import logging
import numpy as np
import os
from typing import List, Tuple, Iterable, Dict
import torch
from src.datasets.dataset_config import DatasetConfig
from src.figures.plot_attention_config import PlotAttnMapConfig
from src.datasets.label_utils import get_unique_parts_from_labels, get_markers_from_labels, get_batches_from_labels
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
from PIL import Image
from matplotlib import gridspec
from tools.load_data_from_npy import parse_paths, load_tile, load_paths_from_npy, Parse_Path_Item
from concurrent.futures import ThreadPoolExecutor


def plot_attn_maps(processed_attn_maps: np.ndarray[float], labels: np.ndarray[str], 
                    paths: np.ndarray[str], data_config: DatasetConfig,  
                    config_plot: PlotAttnMapConfig, output_folder_path: str,
                    num_workers:int = 4, 
                    corr_data = None, corr_method = None):
    """
    for each sample in processed_attn_maps create and saves a figure of the input image, its attention map and overlay. 
    if corr_data is given, adds the corresponding correlation score to the img. 
    """

    os.makedirs(output_folder_path, exist_ok=True)

    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
    
    for i, set_type in enumerate(data_set_types):
        cur_attn_maps, cur_labels, cur_paths = processed_attn_maps[i], labels[i], paths[i]
        if corr_data is not None:
            cur_corr_data = corr_data[i]
        else:
            cur_corr_data = [None]*len(cur_labels)

        logging.info(f'[plot_attn_maps]: for set {set_type}, starting plotting {len(cur_paths)} samples.')
        
        # plot attention samples - multi threading 
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            executor.map(
                lambda args: _plot_single_attn_sample(*args),
                zip(
                    cur_attn_maps,
                    cur_labels,
                    cur_paths,
                    cur_corr_data,
                    [config_plot]*len(cur_attn_maps),
                    [output_folder_path]*len(cur_attn_maps),
                    [corr_method]*len(cur_attn_maps)
                )
            )

def _plot_single_attn_sample(sample_attn, label, img_path, corr, config_plot, output_folder_path, corr_method):
    # load img details
    path_item = parse_paths([img_path]).iloc[0]
    img_path, tile, site = Parse_Path_Item(path_item)
    # plot
    temp_output_folder_path = os.path.join(output_folder_path, os.path.basename(img_path).split('.npy')[0])
    os.makedirs(temp_output_folder_path, exist_ok=True)
    __plot_attn(sample_attn, (img_path, site, tile, label), config_plot, temp_output_folder_path, corr=corr, corr_method=corr_method)

def __plot_attn(proccessed_sample_attn: np.ndarray[float], sample_info:tuple, config_plot, output_folder_path:str, corr:np.ndarray[float] = None, corr_method:str = None):
    """
        calculate correlation data and create figure with the attention map, input image and correlation dta. 
    """

    if proccessed_sample_attn.ndim == 3:
        attn_method = "all_layers"
    elif proccessed_sample_attn.ndim == 2:
        attn_method = "rollout"
    else:
        raise ValueError(f"[plot attn] proccessed_sample_attn shape: {proccessed_sample_attn.shape} is not supported")
        
    globals()[f"_plot_attn_map_{attn_method}"](proccessed_sample_attn, sample_info, config_plot, output_folder_path, corr, corr_method)



def _plot_attn_map_all_layers(processed_attn_map, sample_info, config_plot, output_folder_path, corr = None, corr_method = None):
    # Sample Info
    img_path, site, tile, label = sample_info
    marker, nucleus, input_img = load_tile(img_path, tile)
    assert marker.shape == nucleus.shape
    logging.info(f"[plot_attn_maps] Sample Info: img_path:{img_path}, site:{site}, tile:{tile}, label:{label}")

    # Attn workflow
    num_layers, _, _= processed_attn_map.shape #(num_layers, num_patches, num_patches)
    heatmap_colored_all_layers = []
    for layer_idx in range(num_layers):
        layer_attn = processed_attn_map[layer_idx]
        if corr is not None:
            layer_corr = corr[layer_idx]
        else:
            layer_corr = None
        # create attn map heatmap
        heatmap_colored = __color_heatmap_attn_map(layer_attn, heatmap_color=config_plot.PLOT_HEATMAP_COLORMAP)
        heatmap_colored_all_layers.append(heatmap_colored)

        # plot for each layer seperatly if specified
        if config_plot.SAVE_SEPERATE_LAYERS:
            __create_attn_map_img(layer_attn, input_img, heatmap_colored,config_plot, sup_title =f"Tile{tile}_Layer{layer_idx}\n{label}",  output_folder_path=output_folder_path, corr_data = layer_corr, corr_method = corr_method)
    
    # plot all layers in one figure
    __create_all_layers_attn_map_img(heatmap_colored_all_layers, input_img, config_plot, sup_title = f"Tile{tile}_All_Layers\n{label}", output_folder_path = output_folder_path, corr_data_list = corr, corr_method = corr_method)


def _plot_attn_map_rollout(processed_attn_map, sample_info, config_plot, output_folder_path, corr = None, corr_method = None):
    # Sample Info
    img_path, site, tile, label = sample_info
    marker, nucleus, input_img = load_tile(img_path, tile)
    assert marker.shape == nucleus.shape
    logging.info(f"[plot_attn_maps] Sample Info: img_path:{img_path}, site:{site}, tile:{tile}, label:{label}")

    # Attn workflow
    # create attn map heatmap
    heatmap_colored = __color_heatmap_attn_map(processed_attn_map, heatmap_color=config_plot.PLOT_HEATMAP_COLORMAP)
    
    # create figure
    __create_attn_map_img_test(processed_attn_map, input_img, heatmap_colored, config_plot, sup_title= f"{label}_Tile{tile}", output_folder_path= output_folder_path, corr_data = corr, corr_method = corr_method)



def __create_attn_map_img(attn_map, input_img, heatmap_colored, config_plot, sup_title = "Attention Maps", output_folder_path = None, corr_data = None, corr_method = None):
        """
            Create attention map img with:
                (1) input image 
                (2) attention heatmap
                (3) attention overlay on the input img
            ** save/plot according to config_plot

            parameters:
                attn_map: attention maps values, already in the img shape (H,W), rescale to [0,1]
                input_img: input img with marker and nucleus overlay (3,H,W)
                            ** assuming  Green = nucleus, Blue = marker, Red = zeroed out
                heatmap_colored: attention map colored by heatmap_color (3,H,W)
                config_plot: config with the plotting parameters 
                corr_data: [optional] tuple of corrletion of the attention with the image channels, entropy and corr_method
                sup_title: [optional] main title for the figure
                output_folder_path: [optional] for saving the output fig.

            return:
                fig: matplot fig created. 
        """

        

        alpha = config_plot.ALPHA

        fig, ax = plt.subplots(1, 3, figsize=config_plot.FIG_SIZE)

        if corr_data is not None:
            corr_nucleus, corr_marker = corr_data[0], corr_data[1]
            ax[1].text(0.5, -0.25, f"{corr_method} Correlation (Nucleus): {corr_nucleus:.2f}\n{corr_method} Correlation (Marker): {corr_marker:.2f}",
                    transform=ax[1].transAxes, ha='center', va='center', fontsize=config_plot.PLOT_TITLE_FONTSIZE, color='black')

        
        ax[0].set_title(f'Input - Marker (green), Nucleus (blue)', fontsize=config_plot.PLOT_TITLE_FONTSIZE)
        ax[0].imshow(input_img)
        ax[0].set_axis_off()

        ax[1].set_title(f'Attention Heatmap: No Th', fontsize=config_plot.PLOT_TITLE_FONTSIZE)
        ax[1].imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
        ax[1].set_axis_off()

        fill_cmap = LinearSegmentedColormap.from_list(
            'fill_colors',
            [(0, 0, 0, 0),           # transparent black
            (1, 1, 0, 0.4),         # yellow 
            (1, 0.6, 0, 0.6),       # orange
            (1, 0, 0, 0.8)]         # red
        )

        lines_cmap = LinearSegmentedColormap.from_list(
            'line_colors',
            [(1, 1, 1, 0.2),         # transparent white
            (1, 1, 0, 0.4),         # yellow 
            (1, 0.6, 0, 0.6),       # orange
            (1, 0, 0, 0.8)]         # red
        )

        ax[2].set_title(f'Attention Overlay: Th{config_plot.ATTN_OVERLAY_THRESHOLD}', fontsize=config_plot.PLOT_TITLE_FONTSIZE)
        ax[2].imshow(input_img)  # Show the original image

        levels = np.linspace(config_plot.ATTN_OVERLAY_THRESHOLD, 1.0, config_plot.NUM_CONTOURS) # skip 20% lowest values  
        contours = ax[2].contourf(
            attn_map,
            levels=levels,
            cmap=fill_cmap,
            alpha=alpha,  
        )

        thick_contours = ax[2].contour(
            attn_map,
            levels=levels,              
            cmap=lines_cmap,            
            linewidths=1.0,             
            alpha= alpha + 0.05
        )

        ax[2].set_axis_off()

        fig.suptitle(sup_title, fontsize=config_plot.PLOT_SUPTITLE_FONTSIZE, y=1.1)
        
        if config_plot.SAVE_PLOT and (output_folder_path is not None):
            fig_name  = sup_title.split('\n', 1)[0] #either till the end of the line or the full str
            save_path = os.path.join(output_folder_path, f"{fig_name}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=config_plot.PLOT_SAVEFIG_DPI)
            plt.close()
        if config_plot.SHOW_PLOT:
            plt.show()

        logging.info(f"[plot_attn_maps] attn maps saved: {save_path}")
        return fig




def __create_attn_map_img_test(attn_map, input_img, heatmap_colored, config_plot, sup_title = "Attention Maps", output_folder_path = None, corr_data = None, corr_method = None):
    """
    Visualize attention alongside marker and nucleus channels in a 2x2 layout:

        [0,0] Marker channel       [0,1] Overlay (Input + Attn)
        [1,0] Nucleus channel      [1,1] Heatmap

    Args:
        attn_map: (H, W) attention map, scaled [0, 1]
        input_img: (H, W, 3) RGB image
        heatmap_colored: (H, W, 3) attention heatmap (BGR)
        config_plot: plotting config
        sup_title: overall title
        output_folder_path: optional path to save the figure

    Returns:
        fig: matplotlib figure object
    """

    alpha = config_plot.ALPHA
    
    # Create figure with minimal spacing
    fig, ax = plt.subplots(2, 2, figsize=config_plot.FIG_SIZE, 
                          gridspec_kw={'wspace': 0.02, 'hspace': 0.25})

    # Extract channels
    nucleus = input_img[..., 2]
    marker = input_img[..., 1]

    # Create RGB versions with black background for marker and nucleus
    nucleus_rgb = np.zeros_like(input_img)
    nucleus_rgb[..., 2] = nucleus

    marker_rgb = np.zeros_like(input_img)
    marker_rgb[..., 1] = marker

    # [0,0] Marker channel
    ax[0, 0].imshow(marker_rgb)
    ax[0, 0].set_title("Marker (Green)", fontsize=config_plot.PLOT_TITLE_FONTSIZE, pad=5)
    ax[0, 0].set_axis_off()

    # [0,1] Overlay (Input + Attn)
    ax[0, 1].imshow(input_img)
    ax[0, 1].set_title(f"Overlay (Th={config_plot.ATTN_OVERLAY_THRESHOLD})", fontsize=config_plot.PLOT_TITLE_FONTSIZE, pad=5)

    # Contour fill & lines for attention overlay
    fill_cmap = LinearSegmentedColormap.from_list(
        'fill_colors',
        [(0, 0, 0, 0),
         (1, 1, 0, 0.4),
         (1, 0.6, 0, 0.6),
         (1, 0, 0, 0.8)]
    )
    line_cmap = LinearSegmentedColormap.from_list(
        'line_colors',
        [(1, 1, 1, 0.2),
         (1, 1, 0, 0.4),
         (1, 0.6, 0, 0.6),
         (1, 0, 0, 0.8)]
    )
    levels = np.linspace(config_plot.ATTN_OVERLAY_THRESHOLD, 1.0, config_plot.NUM_CONTOURS)
    ax[0, 1].contourf(attn_map, levels=levels, cmap=fill_cmap, alpha=alpha)
    ax[0, 1].contour(attn_map, levels=levels, cmap=line_cmap, linewidths=1.0, alpha=alpha + 0.05)
    ax[0, 1].set_axis_off()

    # [1,0] Nucleus channel
    ax[1, 0].imshow(nucleus_rgb)
    ax[1, 0].set_title("Nucleus (Blue)", fontsize=config_plot.PLOT_TITLE_FONTSIZE, pad=5)
    ax[1, 0].set_axis_off()

    # [1,1] Heatmap
    ax[1, 1].imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
    ax[1, 1].set_title("Attention Heatmap", fontsize=config_plot.PLOT_TITLE_FONTSIZE, pad=5)
    ax[1, 1].set_axis_off()

    # Add main title with proper spacing
    fig.suptitle(sup_title, fontsize=config_plot.PLOT_SUPTITLE_FONTSIZE, y=0.95)

    # Fine-tune the layout
    plt.subplots_adjust(top=0.88, bottom=0.02, left=0.02, right=0.98)

    if config_plot.SAVE_PLOT and output_folder_path is not None:
        fig_name = sup_title.split('\n', 1)[0]
        save_path = os.path.join(output_folder_path, f"{fig_name}.png")
        plt.savefig(save_path, dpi=config_plot.PLOT_SAVEFIG_DPI, 
                    bbox_inches='tight',facecolor='white', 
                    edgecolor='none', pad_inches=0.05)
        plt.close()
        logging.info(f"[plot_attn_maps] attn maps saved: {save_path}")
    elif config_plot.SHOW_PLOT:
        plt.show()

    return fig



def __create_all_layers_attn_map_img(attn_maps, input_img, config_plot, sup_title = "Attention Maps", output_folder_path = None, corr_data_list = None, corr_method = None):
        """
            Create attention map img with attention maps of all layers
                (1) input image 
                (2) attention heatmaps for all layers
            ** save/plot according to config_plot

            parameters:
                attn_map: attention map colored by heatmap_color (3,H,W) for each layer
                input_img: input img with marker and nucleus overlay (3,H,W)
                            ** assuming  Green = nucleus, Blue = marker, Red = zeroed out
                config_plot: config with the plotting parameters 
                corr_data: [optional] tuple of corrletion of the attention with the image channels, entropy and corr_method
                sup_title: [optional] main title for the figure
                output_folder_path: [optional] for saving the output fig.

            return:
                fig: matplot fig created. 
        """

        fig = plt.figure(figsize=config_plot.ALL_LAYERS_FIG_SIZE, facecolor="#d3ebe3")
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.2)

        # Main title
        fig.suptitle(f"{sup_title}\n\n", fontsize=config_plot.PLOT_SUPTITLE_FONTSIZE, fontweight='bold', y=0.98)

        # Overlay section 
        ax_overlay = plt.subplot(gs[0])
        ax_overlay.imshow(input_img)
        ax_overlay.set_title("Input Image", fontsize=config_plot.PLOT_TITLE_FONTSIZE, fontweight='bold', pad=10)
        ax_overlay.axis("off")

        # Attention maps section 
        gs_attn = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=gs[1], wspace=0.3, hspace=0.8)
        fig.text(0.5, 0.68, "Attention Maps", ha='center', va='center', fontsize=config_plot.PLOT_TITLE_FONTSIZE, fontweight='bold')

        if corr_data_list is None:
            corr_data_list = [None] * len(attn_maps)

        for layer_idx, (attn_map, corr_data) in enumerate(zip(attn_maps, corr_data_list)):

            # plot layer attn maps
            ax = plt.subplot(gs_attn[layer_idx])  
            ax.imshow(cv2.cvtColor(attn_map, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Layer {layer_idx}", fontsize=config_plot.PLOT_TITLE_FONTSIZE, fontweight='bold')
            ax.axis("off")

            # Add correlation values below the attention map
            if corr_data is not None:
                corr_nucleus, corr_marker = corr_data[0], corr_data[1]
                ax.text(0.5, -0.25, f"{corr_method} Correlation (Nucleus): {corr_nucleus:.2f}\n{corr_method} Correlation (Marker): {corr_marker:.2f}", 
                    transform=ax.transAxes, ha='center', va='center', fontsize=config_plot.PLOT_LAYER_FONTSIZE, color='black')
        
        plt.tight_layout()
        if config_plot.SAVE_PLOT and (output_folder_path is not None):
                fig_name  = sup_title.split('\n', 1)[0] #either till the end of the line or the full str
                save_path = os.path.join(output_folder_path, f"{fig_name}.png")
                plt.savefig(save_path, bbox_inches='tight', dpi=config_plot.PLOT_SAVEFIG_DPI)
                plt.close()
        if config_plot.SHOW_PLOT:
                plt.show()

        logging.info(f"[plot_attn_maps] attn maps saved: {save_path}")
        return fig

def __color_heatmap_attn_map(attn_map_img, heatmap_color=cv2.COLORMAP_JET):
    """
    Color an resized and normalized attention map.

    Parameters:
        attn_map_img: float32 NumPy array of shape (H, W), scaled to [0,1]
        heatmap_color: OpenCV colormap type (default: cv2.COLORMAP_JET)

    Returns:
        heatmap_colored: uint8 colored attention heatmap of shape (H, W, 3)
    """
    # Scale to [0, 255] and convert to uint8
    heatmap_uint8 = (attn_map_img * 255).astype(np.uint8)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, heatmap_color)

    return heatmap_colored
