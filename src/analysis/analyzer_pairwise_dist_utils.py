import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
sys.path.insert(0, os.getenv("HOME"))
from NOVA_rotation.load_files.load_data_from_npy import load_npy_to_df, load_npy_to_nparray, load_paths_from_npy
from NOVA.src.datasets.dataset_config import DatasetConfig
from NOVA_rotation.embeddings.embedding_utils.subset_utils import _extract_mutual_params
from NOVA.src.figures.plot_config import PlotConfig
from NOVA.src.common.utils import load_config_file
from NOVA.src.datasets.label_utils import get_batches_from_labels, get_unique_parts_from_labels, get_markers_from_labels, get_batches_from_input_folders, get_cell_lines_conditions_from_labels
import logging


def filter_by_labels(labels_df: pd.DataFrame,embeddings_df: pd.DataFrame,paths_df: pd.DataFrame, filters: dict
):
    """
    Filter labels, embeddings, and paths based on multiple column-value conditions.

    Parameters:
        labels_df (pd.DataFrame): DataFrame containing labels and metadata.
        embeddings_df (pd.DataFrame): DataFrame with embeddings, aligned by index.
        paths_df (pd.DataFrame): DataFrame with file paths or additional info, aligned by index.
        filters (dict): Dictionary of {column_name: value} to filter on.

    Returns:
        filtered_labels, filtered_embeddings, filtered_paths: Filtered DataFrames.
    """
    
    # Apply all filters
    mask = pd.Series(True, index=labels_df.index)
    for col, val in filters.items():
        mask &= (labels_df[col] == val)

    filtered_labels = labels_df[mask]
    filtered_embeddings = embeddings_df.loc[filtered_labels.index]
    filtered_paths = paths_df.loc[filtered_labels.index]

    return filtered_labels, filtered_embeddings, filtered_paths


def compute_pair_wise_distances(a1:np.array, a2:np.array, metric='euclidean'):
    """"Compute all pairwise distances between 2 vectors.
    parameters:
        a1: first array 
        a2: second array
        mteric: metric to calculate the dist accordinly (defualt: cosine)
    
    return:
        dim1: dimension of a1
        dim2: dimension of a2
        flattened_distances: distance matrix 
    """
    from scipy.spatial.distance import cdist
    distances = cdist(a1, a2, metric=metric)  
    return distances

def get_pairs_sectional(pairs, n_pairs, dim1, dim2, without_repeat = True):
    """
        extract pairs with the min/max/middle (sections) distances. 
    return:
        dict: key: section type, value: the pairs - inidices of each "section" pairs 
        ** length of total pairs returned is <= n_pairs * 3

    """

    def unique_pairs(pairs):
        selected = []
        used_i = set()
        used_j = set()
        for i, j in pairs:
            if i not in used_i and j not in used_j:
                selected.append((i, j))
                used_i.add(i)
                used_j.add(j)
                if len(selected) == n_pairs:
                    break
        return selected
    
    # get min/max/middle dist pairs
    middle_start = max(0, (len(pairs) // 2) - (n_pairs // 2))
    middle_end = min(middle_start + n_pairs, len(pairs))
    
    if without_repeat:
        assert len(pairs) >= n_pairs * 3, "[get subset: get pairs] Number of pairs isn't enough for without_repeat."
        min_pairs = unique_pairs(pairs)
        max_pairs = unique_pairs(pairs[::-1])  # reverse for max
        middle_section = pairs[middle_start:middle_end+n_pairs]  # extra in case some repeat
        middle_pairs = unique_pairs(middle_section)
    else:
        min_pairs = pairs[:n_pairs]
        max_pairs = pairs[-n_pairs:]
        middle_pairs = pairs[middle_start:middle_end]
    
    return {"min": min_pairs, "max": max_pairs, "middle":middle_pairs}

def get_pairs_random(pairs, n_pairs, dim1, dim2, without_repeat = True, seed = None):
    """
        extract random pairs. 
    return:
        dict: key: section type, value: the pairs - inidices of each pair
        ** length of total pairs returned is:    <= n_pairs if without_repeat = False
                                                 == n_pairs if without_repeat = True
    """
    if seed is not None:
        np.random.seed(seed)

    if without_repeat:
        # Ensure we have enough unique samples
        max_possible = min(dim1, dim2)
        assert n_pairs <= max_possible, f"[get subset: get pairs] Cannot select {n_pairs} pairs without repeat from {dim1}x{dim2} matrix. Max possible: {max_possible}"
        
        selected = []
        used_i = set()
        used_j = set()
        
        # Shuffle all pairs to randomize selection
        shuffled_pairs = np.random.permutation(pairs)
        
        for i, j in shuffled_pairs:
            if i not in used_i and j not in used_j:
                selected.append((i, j))
                used_i.add(i)
                used_j.add(j)
                if len(selected) == n_pairs:
                    break
    else:
        # Simple random selection with possible repeats
        assert n_pairs <= len(pairs), f"[get subset: get pairs] Cannot select {n_pairs} pairs from {len(pairs)} total pairs"
        selected_indices = np.random.choice(len(pairs), size=n_pairs, replace=False)
        selected =  [pairs[idx] for idx in selected_indices]
    
    return {"random": selected}

def get_pairs(flattened_distances, matrix_shape, config):
    """
        extract pairs using config.SUBSET_METHOD. 

        parameters:
        flattened_distances: flattened distance matrix (m x k)
        matrix_shape: shape of the original matrix
        config: Subset config
    
    return:
        dict: key: section type, value: the pairs - inidices of each "section" pairs 

    """

    sorted_indices = np.argsort(flattened_distances)  # sort (ascendingly)
    pairs = [(idx // matrix_shape[1], idx % matrix_shape[1]) for idx in sorted_indices]  # extract (i, j) from 1D index
    
    labeled_pairs = globals()[f"get_pairs_{config.SUBSET_METHOD}"](pairs, n_pairs = config.NUM_PAIRS, dim1 = matrix_shape[0], dim2 = matrix_shape[1], without_repeat = config.WITHOUT_REPEAT)
    return labeled_pairs


def visualize_pairs(distances, flattened_distances, labeled_pairs, metric, output_dir=None):
    """
    Create a distribution plot of the distances and mark the pairs on top of it.
    
    Parameters:
    distances: 2D distance matrix
    flattened_distances: flattened version of distance matrix
    labeled_pairs: dict with structure {"section_type": [(i, j), ...], ...}
    metric: distance metric name for labeling
    output_dir: directory to save plot (if None, shows plot)
    """
    
    # Color mapping for different pair types
    color_map = {
        'min': 'red',
        'max': 'green', 
        'middle': 'orange',
        'random': 'purple',
        # Add more colors as needed
    }
    
    # Default colors for unknown section types
    default_colors = ['cyan', 'magenta', 'brown', 'pink', 'gray']
    
    plt.figure(figsize=(10, 6))
    plt.hist(flattened_distances, bins=50, alpha=0.7, color='blue', label='All distances')

    # Plot vertical lines for each section type
    legend_info = set()  # Store (section_type, color) for legend creation
    
    for section_type, pairs_list in labeled_pairs.items():
        # Get color for this section type
        if section_type in color_map:
            color = color_map[section_type]
        else:
            # Use default colors for unknown types
            color_idx = len(color_map) % len(default_colors)
            color = default_colors[color_idx]
        
        # Store for legend creation (set automatically handles duplicates)
        legend_info.add((section_type, color))

        # Get distances for this section's pairs
        selected_distances = [distances[i, j] for i, j in pairs_list]
        
        # Plot vertical lines
        for d in selected_distances:
            plt.axvline(d, color=color, linestyle='--', linewidth=1)
    
    for section_type, color in legend_info:
        plt.plot([], [], color=color, linestyle='--', linewidth=2, label=f'{section_type.capitalize()} pairs')

    plt.xlabel(f"{metric} Distance")
    plt.ylabel("Count")
    plt.title("Distribution of All Pairwise Distances")
    plt.legend()

    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{metric}_distance_distribution.png"))
        plt.close()
    else:
        plt.show()


def compute_distances(embeddings:np.ndarray[float], labels:np.ndarray[str], paths: np.ndarray[str], metric:str, data_config):
                """
                extract subset of samples from marker_embeddings by -
                    1) computing all pair-wise distances
                    2) sorting the distances and taking the num_pairs with minimal/middle/maxinal distance
                    3) save csv file with the distances
                    4) extract data (emb, labels, paths) of the curresponding pairs and save them in npy files
                """
                # extract data from config 
                grouped_labels_by_conditions = get_cell_lines_conditions_from_labels(labels, data_config)
                unique_conditions = np.unique(grouped_labels_by_conditions)
                assert len(unique_conditions) == 2, f"[analyzer pairwise dist]: should only have 2 unique conditions! found: {unique_conditions}"
                c1_indices = np.where(grouped_labels_by_conditions == unique_conditions[0])[0]
                c2_indices = np.where(grouped_labels_by_conditions == unique_conditions[1])[0]
                filtered_labels_c1, filtered_embeddings_c1, filtered_paths_c1 = labels[c1_indices], embeddings[c1_indices], paths[c1_indices]
                filtered_labels_c2, filtered_embeddings_c2, filtered_paths_c2 = labels[c2_indices], embeddings[c2_indices], paths[c2_indices]
                                
                # Compute all pairwise distances between condition 1 and 2
                distances = compute_pair_wise_distances(filtered_embeddings_c1, filtered_embeddings_c2, metric)
                
                return distances, unique_conditions, filtered_paths_c1, filtered_paths_c2

def extract_pairs(distances, unique_conditions, paths_c1, paths_c2, config):
                flattened_distances = distances.flatten()
                num_samples_c1, num_samples_c2 = distances.shape
                # extract pairs
                labeled_pairs = get_pairs(flattened_distances, (num_samples_c1, num_samples_c2), config)

                # Get indices for condition 1 and 2 samples
                c1_indices = set()
                c2_indices = set()
                for section in labeled_pairs.keys():
                    # Extract indices and add them individually to sets
                    for i, j in labeled_pairs[section]:
                        c1_indices.add(i)
                        c2_indices.add(j)

                c1_indices = np.array(list(c1_indices))
                c2_indices = np.array(list(c2_indices))

                # Save distances
                distances_data = []
                for section_type, pairs_list in labeled_pairs.items():
                    for i, j in pairs_list:
                        distances_data.append({
                            "pair_type": section_type,
                            f"{config.METRIC}_distance": distances[i, j],
                            f"path_{unique_conditions[0]}": paths_c1[i],
                            f"path_{unique_conditions[1]}": paths_c2[j]
                        })
                distances_df = pd.DataFrame(distances_data)
                return distances_df


def extract_subset(marker_labels:pd.DataFrame, marker_embeddings:pd.DataFrame, marker_paths:pd.DataFrame, set_type:str, data_config:DatasetConfig, output_dir:str):
                """
                extract subset of samples from marker_embeddings by -
                    1) computing all pair-wise distances
                    2) sorting the distances and taking the num_pairs with minimal/middle/maxinal distance
                    3) save csv file with the distances
                    4) extract data (emb, labels, paths) of the curresponding pairs and save them in npy files
                """
                # extract data from config 
                metric:str = data_config.METRIC
                mutual_attr:str = data_config.MUTUAL_ATTR
                compare_by_attr:str = data_config.COMPARE_BY_ATTR
                mutual_param_c1, mutual_param_c2 = _extract_mutual_params(getattr(data_config ,data_config.MUTUAL_ATTR))
                compare_param_c1 = getattr(data_config, data_config.COMPARE_BY_ATTR)[0]
                compare_param_c2 = getattr(data_config, data_config.COMPARE_BY_ATTR)[1]

                filtered_labels_c1, filtered_embeddings_c1, filtered_paths_c1  = filter_by_labels(marker_labels, marker_embeddings, marker_paths, {mutual_attr.lower():mutual_param_c1, compare_by_attr.lower():compare_param_c1})
                filtered_labels_c2, filtered_embeddings_c2, filtered_paths_c2 = filter_by_labels(marker_labels, marker_embeddings, marker_paths, {mutual_attr.lower():mutual_param_c2, compare_by_attr.lower():compare_param_c2})
                
                #convert to nparraay
                filtered_embeddings_c1 = np.array(filtered_embeddings_c1)
                filtered_embeddings_c2 = np.array(filtered_embeddings_c2)
                
                num_samples_c1, num_samples_c2 = len(filtered_embeddings_c1), len(filtered_embeddings_c2)
                
                # Compute all pairwise distances between condition 1 and 2
                distances = compute_distances(filtered_embeddings_c1, filtered_embeddings_c2, metric)
                flattened_distances = distances.flatten()
                
                # extract min/max/middle pairs
                labeled_pairs = get_pairs(flattened_distances, (num_samples_c1, num_samples_c2), data_config)

                # Get indices for condition 1 and 2 samples
                c1_indices = set()
                c2_indices = set()
                for section in labeled_pairs.keys():
                    # Extract indices and add them individually to sets
                    for i, j in labeled_pairs[section]:
                        c1_indices.add(i)
                        c2_indices.add(j)

                c1_indices = np.array(list(c1_indices))
                c2_indices = np.array(list(c2_indices))
                logging.info(f"Selected {len(c1_indices)} {mutual_param_c1}:{compare_param_c1} samples and {len(c2_indices)} {mutual_param_c2}:{compare_param_c2} samples.")

                # Save distances
                distances_data = []
                for section_type, pairs_list in labeled_pairs.items():
                    for i, j in pairs_list:
                        distances_data.append({
                            "pair_type": section_type,
                            f"{metric}_distance": distances[i, j],
                            f"path_{compare_param_c1}": filtered_paths_c1.iloc[i]["Path"],
                            f"path_{compare_param_c2}": filtered_paths_c2.iloc[j]["Path"]
                        })
                distances_df = pd.DataFrame(distances_data)
                distances_df.to_csv(os.path.join(output_dir, f"{metric}_distances.csv"), index=False)

                # extract embeding,labels and paths values
                c1_embeddings = filtered_embeddings_c1[c1_indices]
                c1_labels = filtered_labels_c1.iloc[c1_indices]
                c1_paths = filtered_paths_c1.iloc[c1_indices]

                c2_embeddings = filtered_embeddings_c2[c2_indices]
                c2_labels = filtered_labels_c2.iloc[c2_indices]
                c2_paths = filtered_paths_c2.iloc[c2_indices]

                set_embeddings = np.concatenate([c1_embeddings, c2_embeddings], axis=0)
                set_labels = pd.concat([c1_labels, c2_labels], axis=0).reset_index(drop=True)
                set_paths= pd.concat([c1_paths, c2_paths], axis=0).reset_index(drop=True)

                # Save npy files
                np.save(os.path.join(output_dir, f"{set_type}.npy"), set_embeddings)
                np.save(os.path.join(output_dir, f"{set_type}_labels.npy"), set_labels["full_label"].values)
                np.save(os.path.join(output_dir, f"{set_type}_paths.npy"), np.array(set_paths["Path"].values, dtype=str))
   
                visualize_pairs(distances, flattened_distances, labeled_pairs, metric, output_dir=output_dir)

