import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.figures.plotting_utils import save_plot, FONT_PATH
from src.datasets.dataset_config import DatasetConfig
from src.figures.plot_config import PlotConfig
from src.datasets.label_utils import get_markers_from_labels, get_unique_parts_from_labels, map_labels
from src.common.utils import get_if_exists, save_config

import logging
import numpy as np
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

import matplotlib
from matplotlib import font_manager as fm

fm.fontManager.addfont(FONT_PATH)
matplotlib.rcParams['font.family'] = 'Arial'

def plot_umap(umap_embeddings: np.ndarray[float], labels: np.ndarray[str], config_data: DatasetConfig,
              config_plot: PlotConfig, saveroot: str, umap_idx: int, ari_scores:Dict[str,float]) -> None:
    """Unified function to plot 2D UMAP embeddings with different modes.

    Args:
        umap_embeddings (np.ndarray[float]): The 2D UMAP embeddings.
        labels (np.ndarray[str]): Array of labels corresponding to the embeddings.
        config_data (DatasetConfig): Configuration data containing visualization settings.
        saveroot (str): Root path to save the plot and configuration.
        umap_idx (int): UMAP type index to distinguish between modes (0: individual markers, 1: all markers, 2: concatenated embeddings).
        ari_scores (Dict): A dictionary with ari values

    Raises:
        ValueError: If an invalid `umap_idx` is provided.
    """

    if saveroot:
        os.makedirs(saveroot, exist_ok=True)
        save_config(config_data, saveroot)
        save_config(config_plot, saveroot)

    if umap_idx == 0:
        # Mode: Individual markers
        markers = get_unique_parts_from_labels(labels, get_markers_from_labels)
        logging.info(f"[plot_umap] Detected markers: {markers}")
        for marker in markers:
            logging.info(f"[plot_umap]: Marker: {marker}")
            indices = np.where(np.char.startswith(labels.astype(str), f"{marker}_"))[0]
            logging.info(f"[plot_umap]: {len(indices)} indexes have been selected")

            if len(indices) == 0:
                logging.info(f"[plot_umap] No data for marker {marker}, skipping.")
                continue

            marker_umap_embeddings = umap_embeddings[indices]
            marker_labels = labels[indices].reshape(-1,)

            savepath = os.path.join(saveroot, f'{marker}') if saveroot else None
            label_data = map_labels(marker_labels, config_plot, config_data)

            if config_data.SHOW_ARI:
                ari_score = ari_scores[marker]
            else:
                ari_score = None
            __plot_umap_embeddings(marker_umap_embeddings, label_data, config_data, config_plot, savepath=savepath, title=marker,
                                   ari_score=ari_score)
        return

    elif umap_idx == 1:
        # Mode: All markers together
        savepath = os.path.join(saveroot, 'umap1') if saveroot else None
    elif umap_idx == 2:
        # Mode: Concatenated embeddings
        savepath = os.path.join(saveroot, 'umap2') if saveroot else None
    
    label_data = map_labels(labels, config_plot, config_data)
    if config_data.SHOW_ARI:
            ari_score = ari_scores['ari']
    else:
        ari_score = None
    __plot_umap_embeddings(umap_embeddings, label_data, config_data, config_plot, savepath, ari_score=ari_score)

    
def __get_metrics_figure(score:float, ax:Axes=None)->Axes:
    """Generate a plot displaying the metrics

    Args:
        score (float): The calculated score to be plotted
        ax (Axes, optional): The Axes object to add the metric to. Defaults to None.

    Returns:
        Axes: The figure
    """
    title = "ARI"
    vrange = (-0.5,1)

    # Optional
    linecolor = 'r'
    linewidth = 5
    cmap = "Greys"

    if ax is None:
        _, ax = plt.subplots(1,1, figsize=(5,0.2))

    vmin, vmax = vrange[0], vrange[1]
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(sm, cax=ax, orientation='horizontal', pad=0.25)
    cb.set_ticks([score])
    cb.ax.set_title(title)
    cb.ax.plot([score]*2, [vmin,vmax], linecolor, linewidth=linewidth)

    return ax

def __plot_umap_embeddings(umap_embeddings: np.ndarray[float], 
                         label_data: np.ndarray[str], 
                         config_data: DatasetConfig,
                         config_plot: PlotConfig,
                         savepath: str = None,
                         title: str = None, 
                         dpi: int = 500, 
                         figsize: Tuple[int,int] = (6,5),
                         cmap:str = 'tab20',
                         ari_score:float = None,
                         ) -> None:
    """Plots UMAP embeddings with given labels and configurations.

    Args:
        umap_embeddings (np.ndarray[float]): The 2D UMAP embeddings to be plotted.
        label_data (np.ndarray[str]): Array of labels corresponding to the embeddings.
        config_data (DatasetConfig): Configuration data containing metric settings.
        config_plot (PlotConfig): Configuration plot containing visualization settings.
        savepath (str, optional): Path to save the plot. If None, the plot is shown interactively. Defaults to None.
        title (str, optional): Title for the plot. Defaults to 'UMAP projection of Embeddings'.
        dpi (int, optional): Dots per inch for the saved plot. Defaults to 300.
        figsize (Tuple[int, int], optional): Size of the figure. Defaults to (6, 5).
        cmap (str, optional): Colormap to be used. Defaults to 'tab20'.
        ari_score (float, optional): ari score to show on the umap. Defaults to None.

    Raises:
        ValueError: If the size of `umap_embeddings` and `label_data` are incompatible.

    Returns:
        None
    """
    if umap_embeddings.shape[0] != label_data.shape[0]:
        raise ValueError("The number of embeddings and labels must match.")

    name_color_dict =  config_plot.COLOR_MAPPINGS
    name_key = config_plot.UMAP_MAPPINGS_ALIAS_KEY
    color_key = config_plot.UMAP_MAPPINGS_COLOR_KEY
    marker_size = config_plot.SIZE
    alpha = config_plot.ALPHA
    to_color = get_if_exists(config_plot, 'TO_COLOR', None)
    show_metric = config_data.SHOW_ARI
    
    unique_groups = np.unique(label_data)

    ordered_marker_names = get_if_exists(config_plot, 'ORDERED_MARKER_NAMES', None)
    if ordered_marker_names:
        # Get the indices of each element in 'unique_groups' according to 'ordered_marker_names'
        indices = [ordered_marker_names.index(item) for item in unique_groups]
        # Sort the unique_groups based on the indices
        unique_groups = unique_groups[np.argsort(indices)]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2,1,height_ratios=[20,1])

    ax = fig.add_subplot(gs[0])
    for i, group in enumerate(unique_groups):
        logging.info(f'[_plot_umap_embeddings]: adding {group}')
        indices = np.where(label_data==group)[0]
        if group == 'DAPI':
            np.random.seed(config_plot.SEED)
            indices = np.random.choice(indices, size=int(len(indices) * 0.1), replace=False)
        # Get hex color and convert to RGBA
        if to_color is not None and group not in to_color:
            base_color = '#bab5b5'
            alpha = 0.4
        else:
            base_color = name_color_dict[group][color_key] if name_color_dict else plt.get_cmap(cmap)(i)

        rgba_color = mcolors.to_rgba(base_color, alpha=alpha)  # Convert hex to RGBA and apply alpha
        
        # Create a color array for each point
        color_array = np.array([rgba_color] * indices.shape[0])

        label = name_color_dict[group][name_key] if name_color_dict else group

        ax.scatter(
            umap_embeddings[indices, 0],
            umap_embeddings[indices, 1],
            s=marker_size,
            alpha=alpha,
            c=color_array,
            marker = 'o',
            label=label,
            linewidths=0,
        )
        logging.info(f'[_plot_umap_embeddings]: adding label {label}')
        
    __format_UMAP_axes(ax, title)
    __format_UMAP_legend(ax, marker_size)
        
    if show_metric:
        gs_bottom = fig.add_subplot(gs[1])
        ax = __get_metrics_figure(ari_score, ax=gs_bottom)
    
    fig.tight_layout()
    
    if savepath:
        save_plot(fig, savepath, dpi, save_eps=True)
    else:
        plt.show()
        
    return

def __format_UMAP_axes(ax:Axes, title:str)->None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title(title)
    
    ax.set_xticklabels([]) 
    ax.set_yticklabels([]) 
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    return

def __format_UMAP_legend(ax:Axes, marker_size: int) -> None:
    """Formats the legend in the plot."""
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, prop={'size': 6},
                    bbox_to_anchor=(1, 1), loc='upper left',
                    ncol=1 + len(labels) // 26, frameon=False)
    for handle in leg.legendHandles:
        handle.set_alpha(1)
        handle.set_sizes([max(6, marker_size)])