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
from tools.attn_maps_plotting.plot_attention_config import PlotAttnMapConfig
from src.datasets.label_utils import get_unique_parts_from_labels, get_markers_from_labels, get_batches_from_labels
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from matplotlib import gridspec
from tools.load_data_from_npy import parse_paths, load_tile, load_paths_from_npy, parse_path_item
from concurrent.futures import ThreadPoolExecutor
from matplotlib.colors import LinearSegmentedColormap


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
    img_path, tile, site = parse_path_item(path_item)
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
    logging.info(f"[plot_attn_maps] Sample Path: {img_path}")
    logging.info(f"[plot_attn_maps] Sample Info: site:{site}, tile:{tile}, label:{label}")

    # Attn workflow
    # create attn map heatmap
    heatmap_colored = __color_heatmap_attn_map(processed_attn_map, heatmap_color=config_plot.PLOT_HEATMAP_COLORMAP)
    
    # create figure
    __create_attn_map_img(processed_attn_map, input_img, heatmap_colored, config_plot, sup_title= f"{label}_Tile{tile}", output_folder_path= output_folder_path, corr_data = corr, corr_method = corr_method)



def __create_attn_map_img(attn_map, input_img, heatmap_colored, config_plot, 
                          sup_title="Attention Maps", output_folder_path=None, 
                          corr_data=None, corr_method=None):
    """
    Visualize attention maps with flexible component selection.

    Supported components in config_plot.FIG_COMPONENTS:
        - "Marker"   : Marker channel
        - "Overlay"  : Overlay (Input + Attn)
        - "Nucleus"  : Nucleus channel
        - "Heatmap"  : Attention heatmap
    (displayed by the order in the list)

    Args:
        attn_map: (H, W) attention map, scaled [0, 1]
        input_img: (H, W, 3) RGB image
        heatmap_colored: (H, W, 3) attention heatmap (BGR)
        config_plot: plotting config
        sup_title: overall title
        output_folder_path: optional path to save the figure
        corr_data:tuple (optional): attn score tuple for the sample. displayed if given.
        corr_method:str (optional): attn score method name. displayed if corr_data is given.

    Returns:
        fig: matplotlib figure object
    """

    alpha = config_plot.ALPHA
    components = config_plot.FIG_COMPONENTS

    def plot_marker(ax):
        marker = input_img[..., 1]
        marker_rgb = np.zeros_like(input_img)
        marker_rgb[..., 1] = marker
        ax.imshow(marker_rgb)
        if config_plot.DISPLAY_COMPONENTS_TITLE:
            ax.set_title("Marker", fontsize=config_plot.PLOT_TITLE_FONTSIZE, pad=5)
        ax.set_axis_off()
        if corr_data is not None:
            corr_marker = corr_data[1]
            formatted = ", ".join(f"{v:.3f}" for v in corr_marker)
            ax.text(0.5, -0.05, f"{corr_method} Score:\n{formatted}",
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=config_plot.PLOT_TITLE_FONTSIZE, color='black')

    def plot_overlay(ax):
        logging.info(f"[plot_attn_maps] Attention overlay threshold: {config_plot.ATTN_OVERLAY_THRESHOLD}")
        ax.imshow(input_img)
        if config_plot.DISPLAY_COMPONENTS_TITLE:
            ax.set_title(f"Overlay", 
                     fontsize=config_plot.PLOT_TITLE_FONTSIZE, pad=5)
        levels = np.linspace(config_plot.ATTN_OVERLAY_THRESHOLD, 1.0, config_plot.NUM_CONTOURS)

        fill_cmap = LinearSegmentedColormap.from_list(
            'fill_colors',
            config_plot.FILL_CMAP_LIST
        )
 
        line_cmap = LinearSegmentedColormap.from_list(
            'line_colors',
            config_plot.LINE_CMAP_LIST
        )
        ax.contourf(attn_map, levels=levels, cmap=fill_cmap, alpha=alpha)
        ax.contour(attn_map, levels=levels, cmap=line_cmap,
                   linewidths=1.0, alpha=alpha + 0.05)
        ax.set_axis_off()

    def plot_nucleus(ax):
        nucleus = input_img[..., 2]
        nucleus_rgb = np.zeros_like(input_img)
        nucleus_rgb[..., 2] = nucleus
        ax.imshow(nucleus_rgb)

        if config_plot.DISPLAY_COMPONENTS_TITLE:
            ax.set_title("Nucleus", fontsize=config_plot.PLOT_TITLE_FONTSIZE, pad=5)

        ax.set_axis_off()
        if corr_data is not None:
            corr_nucleus = corr_data[0]
            formatted = ", ".join(f"{v:.3f}" for v in corr_nucleus)
            ax.text(0.5, -0.05, f"{corr_method} Score:\n{formatted}",
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=config_plot.PLOT_TITLE_FONTSIZE, color='black')

    def plot_heatmap(ax):
        ax.imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
        if config_plot.DISPLAY_COMPONENTS_TITLE:
            ax.set_title("Attention Heatmap", fontsize=config_plot.PLOT_TITLE_FONTSIZE, pad=5)
        ax.set_axis_off()

    component_map = {
        "Marker": plot_marker,
        "Overlay": plot_overlay,
        "Nucleus": plot_nucleus,
        "Heatmap": plot_heatmap,
    }

    # Build dynamic grid based on selected components
    n_components = len(components)
    ncols = 2 if n_components > 1 else 1
    nrows = int(np.ceil(n_components / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=config_plot.FIG_SIZE,
                             squeeze=False,
                             gridspec_kw={'wspace': 0.02, 'hspace': 0.25})

    axes = axes.flatten()

    for ax, comp in zip(axes, components):
        component_map[comp](ax)

    # Hide any unused axes
    for ax in axes[len(components):]:
        ax.axis("off")

    if config_plot.DISPLAY_SUPTITLE:
        fig.suptitle(sup_title, fontsize=config_plot.PLOT_SUPTITLE_FONTSIZE, y=0.98)
    plt.subplots_adjust(top=0.88, bottom=0.02, left=0.02, right=0.98)

    if config_plot.SAVE_PLOT and output_folder_path is not None:
        fig_name = sup_title.split('\n', 1)[0]
        save_path = os.path.join(output_folder_path, f"{fig_name}.png")
        plt.savefig(save_path, dpi=config_plot.PLOT_SAVEFIG_DPI,
                    bbox_inches='tight', facecolor='white',
                    edgecolor='none', pad_inches=0.05)
        logging.info(f"[plot_attn_maps] attn maps saved: {save_path}")
    
    if config_plot.SHOW_PLOT:
        plt.show()
        
    plt.close()
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
