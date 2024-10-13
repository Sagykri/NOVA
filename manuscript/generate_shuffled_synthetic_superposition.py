import os
import sys


sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

import logging
import numpy as np
from src.common.utils import load_config_file, get_if_exists
from src.embeddings.embeddings_utils import load_embeddings
from src.figures.umap_plotting import plot_umap
from src.datasets.dataset_config import DatasetConfig
from src.figures.plot_config import PlotConfig
from src.analysis.analyzer_umap_multiplex_markers import AnalyzerUMAPMultiplexMarkers
from src.datasets.label_utils import split_markers_from_labels


def generate_shuffled_synthetic_superposition_umap(output_folder_path:str, config_path_data:str, config_path_plot:str)->None:
    """Generating a synthetic superposition after shuffling the phenotypes to create incorrect matches and serve as a sanity check of the method
    """
    config_data:DatasetConfig = load_config_file(config_path_data, 'data')
    config_data.OUTPUTS_FOLDER = output_folder_path
    config_plot:PlotConfig = load_config_file(config_path_plot, 'plot')
    umap_idx = get_if_exists(config_plot, 'UMAP_TYPE', None)
    
    assert umap_idx == analyzer_UMAP.UMAPType['MULTIPLEX_MARKERS'], "plot configuration is not set for MULTIPLEX_MARKERS umap"
    
    embeddings, labels = load_embeddings(output_folder_path, config_data)
    
    # Shuffle phenotypes within the labels while keeping the markers identity
    labels = __get_shuffled_labels_by_phenotype(labels, mismatch_threshold=0.95, seed=config_data.SEED)

    analyzer_UMAP = AnalyzerUMAPMultiplexMarkers(config_data, output_folder_path)
    umap_embeddings, labels, ari_scores = analyzer_UMAP.calculate(embeddings, labels)

    # Define the output folder path and plot the UMAP
    saveroot = analyzer_UMAP.get_saving_folder(feature_type = os.path.join('UMAPs', f'{analyzer_UMAP.UMAPType(umap_idx).name}_shuffled'))  
    plot_umap(umap_embeddings, labels, config_data, config_plot, saveroot, umap_idx, ari_scores)
        
def __get_shuffled_labels_by_phenotype(labels:np.ndarray[str], mismatch_threshold:float=0.95, seed:int=1)->np.ndarray[str]:
    """Get the phenotypes of the labels shuffled with each other while keeping the markers identity at place

    Args:
        labels (np.ndarray[str]): The labels
        mismatch_threshold (float, optional): Minimum percentage for mismatches between the shuffled and the original labels. Defaults to 0.95.
        seed (int, optional): The seed. Defaults to 1.

    Returns:
        np.ndarray[str]: The shuffled labels
    """
    
    logging.info(f"Labels before shuffling: {labels}")

    # Split the marker from the phenotype
    marker_labels, phenotype_labels = split_markers_from_labels([labels], None)

    # Shuffle the phenotypes
    phenotype_labels = __shuffle_labels(phenotype_labels, mismatch_threshold=mismatch_threshold, seed=seed)

    # Reconstruct the labels
    shuffled_labels = np.array([f"{marker}_{phenotype}" for marker, phenotype in zip(marker_labels, phenotype_labels)])
    logging.info(f"Labels after shuffling: {labels}")

    # Since labels might keep their origianl place after shuffling, we want to see how many did change their position
    mismatches_count = len(np.where(shuffled_labels != labels)[0])
    logging.info(f"%Mismatches: {mismatches_count * 100.0 / len(labels)}%")
    
    return shuffled_labels
        
def __shuffle_labels(labels: np.ndarray[str], mismatch_threshold:float=0.95, seed:int=1)->np.ndarray[str]:
    """Shuffle the given labels until reaching the threshold for mismatches between the shuffled and the original labels

    Args:
        labels (np.ndarray[str]): The labels to shuffle
        mismatch_threshold (float, optional): Minimum percentage for mismatches between the shuffled and the original labels. Defaults to 0.95.
        seed (int, optional): The seed. Defaults to 1.

    Returns:
        np.ndarray[str]: The shuffled labels
    """
    assert 0<mismatch_threshold<=1, "mismatch_threshold must be a value between 0 to 1 (included)"
    
    shuffled_labels:np.ndarray[str] = labels.copy()  
    
    # Apply the seed
    np.random.seed(seed)

    # Shuffle the whole array initially
    np.random.shuffle(shuffled_labels)

    # Set the threshold for the number of mismatches
    threshold = int(len(labels) * mismatch_threshold)
    logging.info(f"Mismatch threshold is {threshold}")
    
    # Keep track of which elements are in the different position
    mismatches = labels != shuffled_labels

    # Continue shuffling only the mismatching positions
    while np.sum(mismatches) <= threshold:
        # Find the indices where the label changed
        mismatching_indices = np.where(mismatches)[0]
        
        # Shuffle only the labels at the mismatching positions
        shuffled_subset = shuffled_labels[mismatching_indices].copy()  
        np.random.shuffle(shuffled_subset)  
        shuffled_labels[mismatching_indices] = shuffled_subset
        
        # Recompute mismatches after shuffling
        mismatches = labels != shuffled_labels

    return shuffled_labels


if __name__ == "__main__":
    print("Starting generating shuffled synthetic superposition umap...")
    try:
        if len(sys.argv) < 4:
            raise ValueError("Invalid arguments. Must supply output folder path, data config and plot config.")
        output_folder_path = sys.argv[1]
        config_path_data = sys.argv[2]
        config_path_plot = sys.argv[3]

        generate_shuffled_synthetic_superposition_umap(output_folder_path, config_path_data, config_path_plot)

    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
