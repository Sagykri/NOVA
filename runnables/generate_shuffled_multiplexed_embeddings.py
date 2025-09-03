import os
import sys


sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

import logging
import numpy as np
from src.common.utils import load_config_file, get_if_exists
from src.embeddings.embeddings_utils import load_embeddings
from src.datasets.dataset_config import DatasetConfig
from src.embeddings.embeddings_utils import generate_multiplexed_embeddings, save_embeddings
from src.datasets.label_utils import split_markers_from_labels


def generate_shuffled_multiplexed_embeddings_with_config(outputs_folder_path:str, config_path_data:str)->None:
    config_data:DatasetConfig = load_config_file(config_path_data, "data")
    config_data.OUTPUTS_FOLDER = outputs_folder_path

    # generate multiplex embeddings (using already generated single-marker embeddings)
    # Shuffle phenotypes within the labels while keeping the markers identity
    embeddings, labels, paths = generate_multiplexed_embeddings(outputs_folder_path, config_data,
                                                                format_labels_func= lambda labels: __get_shuffled_labels_by_phenotype(labels, match_threshold=0.05, seed=config_data.SEED))
  
    # save multiplex vectors
    save_embeddings(embeddings, labels, paths, config_data, outputs_folder_path, multiplex = True, folder_name_postfix='_shuffled')
        
def __get_shuffled_labels_by_phenotype(labels:np.ndarray[str], match_threshold:float=0.05, seed:int=1)->np.ndarray[str]:
    """Get the phenotypes of the labels shuffled with each other while keeping the markers identity at place

    Args:
        labels (np.ndarray[str]): The labels
        match_threshold (float, optional): Maximum percentage for matches between the shuffled and the original labels. Defaults to 0.05.
        seed (int, optional): The seed. Defaults to 1.

    Returns:
        np.ndarray[str]: The shuffled labels
    """
    
    logging.info(f"Labels before shuffling: {labels}")

    # Split the marker from the phenotype
    marker_labels, phenotype_labels = split_markers_from_labels(labels, None)
    
    # Shuffle the phenotypes
    phenotype_labels = __shuffle_labels(phenotype_labels, match_threshold=match_threshold, seed=seed)

    # Reconstruct the labels
    shuffled_labels = np.array([f"{marker}_{phenotype}" for marker, phenotype in zip(marker_labels, phenotype_labels)])
    logging.info(f"Labels after shuffling: {shuffled_labels}")

    # Since labels might keep their origianl place after shuffling, we want to see how many kept their original position
    matches_count = len(np.where(shuffled_labels == labels)[0])
    logging.info(f"%Matches: {matches_count * 100.0 / len(labels)}%")
    
    return shuffled_labels
        
def __shuffle_labels(labels: np.ndarray[str], match_threshold:float=0.05, seed:int=1)->np.ndarray[str]:
    """Shuffle the given labels until reaching the threshold for allowed maximum matches between the shuffled and the original labels

    Args:
        labels (np.ndarray[str]): The labels to shuffle
        match_threshold (float, optional): Maximum percentage for matches between the shuffled and the original labels. Defaults to 0.05.
        seed (int, optional): The seed. Defaults to 1.

    Returns:
        np.ndarray[str]: The shuffled labels
    """
    assert 0<=match_threshold<1, "match_threshold must be a value between 0 (included) to 1"
    
    shuffled_labels:np.ndarray[str] = labels.copy()  
    
    # Apply the seed
    np.random.seed(seed)

    # Shuffle the whole array initially
    np.random.shuffle(shuffled_labels)

    # Set the threshold for the number of maximum allowed matches
    threshold = int(len(labels) * match_threshold)
    logging.info(f"Match threshold is {threshold}")
    
    # Keep track of which elements are still in the same position
    matches = labels == shuffled_labels

    # Continue shuffling only the matching positions
    while np.sum(matches) > threshold:
        # Find the indices where the label hasn't changed
        matching_indices = np.where(matches)[0]
        
        # Shuffle only the labels at the matching positions
        shuffled_subset = shuffled_labels[matching_indices].copy()  
        np.random.shuffle(shuffled_subset)  
        shuffled_labels[matching_indices] = shuffled_subset
        
        # Recompute matches after shuffling
        matches = labels == shuffled_labels

    return shuffled_labels


if __name__ == "__main__":
    print("Starting generate shuffled multiplex embeddings...")
    try:
        if len(sys.argv) < 3:
            raise ValueError("Invalid arguments. Must supply outputs folder path and data config.")
        outputs_folder_path = sys.argv[1]
        config_path_data = sys.argv[2]

        generate_shuffled_multiplexed_embeddings_with_config(outputs_folder_path, config_path_data)
        
    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
