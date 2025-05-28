import os
import re
import json
import sys
import subprocess
import datetime
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import Counter
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split

# --- Utility function to parse a label string into structured fields ---
def parse_label(label: str) -> dict:
    """
    Parse a string label into its structured components.

    Args:
        label (str): A string of the format 'marker_line_condition_batch_rep'.

    Returns:
        dict: A dictionary containing keys 'marker', 'line', 'condition', 'batch', and 'rep'.
    """
    parts = label.split('_')
    return {
        'marker': parts[0],
        'line': parts[1],
        'condition': parts[2],
        'batch': parts[3] if len(parts) > 3 else None,
        'rep': parts[4] if len(parts) > 4 else None
    }

def compute_pairwise_distances_and_masks(embeddings: np.ndarray, labels: List[str], pos_keys: List[str], dists:np.ndarray=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pairwise distances and extract the positive and negative distance vectors based on label similarity.

    Args:
        embeddings (np.ndarray): Embedding vectors.
        labels (List[str]): Corresponding labels.
        pos_keys (List[str]): Keys for determining positive relationships.
        dists (np.ndarray, optional): Precomputed pairwise distances. If None, will compute distances.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: All distances, positive distances, negative distances, positive mask.
    """

    N = embeddings.shape[0]

    parsed = [parse_label(l) for l in labels]
    
    # Create a boolean mask for positive pairs based on the specified keys
    pos_mask = np.ones((N, N), dtype=bool)
    for key in pos_keys:
        attr = np.array([p[key] for p in parsed])
        pos_mask &= np.equal.outer(attr, attr)

    # Ensure the diagonal is False (no self-comparisons)
    np.fill_diagonal(pos_mask, False)

    if dists is None:
        dists = pairwise_distances(embeddings, metric='euclidean')

    pos_dists = dists[pos_mask]
    neg_dists = dists[~pos_mask]

    return dists, pos_dists, neg_dists, pos_mask

def compute_alignment(pos_dists:np.ndarray, labels: List[str], alpha: float = 2) -> float:
    """ Compute alignment and uniformity metrics based on positive and negative distances.
    Args:
        pos_dists (np.ndarray): Array of positive distances.
        labels (List[str]): List of label strings.
        alpha (float): Exponent for alignment calculation.
    Returns:
        float: Alignment metric.
    """

    # Calculate alignment on positives
    alignment = np.mean(pos_dists ** alpha) if len(pos_dists) > 0 else np.nan

    return alignment

def compute_uniformity(neg_dists: np.ndarray, labels: List[str], t: float = 2.0) -> float:
    """
    Compute uniformity metric based on negative distances.
    Args:
        neg_dists (np.ndarray): Array of negative distances.
        labels (List[str]): List of label strings.
        t (float): Temperature parameter for uniformity calculation.
    Returns:
        float: Uniformity metric.
    """

    # Calculate uniformity on negatives
    uniformity = np.log(np.mean(np.exp(-t * neg_dists))) if len(neg_dists) > 0 else np.nan

    return uniformity

def compute_recall_at_k(distances:np.ndarray, labels: List[str], k=5):
    """
    Compute recall@k metric.

    Args:
        distances (np.ndarray): Pairwise distance matrix.
        labels (List[str]): raw label strings
        k (int): value of K for recall

    Returns:
        float: recall@k_score
    """

    N = len(labels)

    recall_total = 0
    for i in range(N):
        distances[i, i] = np.inf
        nearest = np.argsort(distances[i])[:k]
        correct = sum(labels[i] == labels[j] for j in nearest)
        recall_total += correct / k

    recall_at_k = recall_total / N
    return recall_at_k

def compute_negative_neighbor_entropy(distances: np.ndarray, pos_mask: np.ndarray, labels: List[str], k: int = 5) -> float:
    """
    Compute the entropy of the top-k negative neighbors for each embedding.

    Args:
        distances (np.ndarray): Pairwise distance matrix.
        pos_mask (np.ndarray): Boolean mask indicating positive neighbors.
        labels (List[str]): List of label strings.
        k (int): Number of top neighbors to consider.

    Returns:
        float: Average entropy of top-k negative neighbors across all embeddings.
    """
    entropies = []

    logging.info(f"Computing negative neighbor entropy@{k}")
    N = len(labels)
    
    for i in range(N):
        # Get indices of negative neighbors
        neg_indices = np.where(~pos_mask[i])[0]
        if len(neg_indices) < k:
            continue

        # Get distances to negative neighbors 
        neg_dists = distances[i][neg_indices]

        # Sort and select top-k negative neighbors
        top_k = np.argsort(neg_dists)[:k]
        top_k_labels = [labels[neg_indices[j]] for j in top_k]

        # Calculate entropy of the top-k negative neighbors
        freq = Counter(top_k_labels)
        probs = np.array([v / k for v in freq.values()])
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        entropies.append(entropy)

    return float(np.mean(entropies)) if entropies else 0.0

def normalize(x: np.ndarray) -> np.ndarray:
    """
    Perform L2 normalization on the last axis of the input array.

    Args:
        x (np.ndarray): Input array of shape (N, D).

    Returns:
        np.ndarray: L2-normalized array of the same shape.
    """
    norm = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    return x / np.where(norm == 0, 1, norm)

def load_model_data(model_dir: str, experiment_type: str, batches: List[str]) -> Tuple[np.ndarray, List[str]]:
    """ Load model embeddings and labels for a specific experiment type and batches.
    Args:
        model_dir (str): Directory containing model folders.
        experiment_type (str): Type of experiment (e.g., 'neurons').
        batches (List[str]): List of batch names to load.
    Returns:
        Tuple[np.ndarray, List[str]]: Tuple containing all embeddings and corresponding labels.
    """
    logging.info(f"Loading model data for experiment: {experiment_type}, batches: {batches}")

    all_embeddings = []
    all_labels = []

    for batch in batches:
        try:
            emb_path = os.path.join(model_dir, 'embeddings', experiment_type, batch, 'testset.npy')
            label_path = os.path.join(model_dir, 'embeddings', experiment_type, batch, 'testset_labels.npy')

            #!!!! NORMALIZING THE EMBEDDINGS!!!!
            logging.warning("NOTE: Normalizing embeddings!!!")
            embeddings = normalize(np.load(emb_path))

            labels = np.load(label_path).tolist()
            assert embeddings.shape[0] == len(labels)
            all_embeddings.append(embeddings)
            all_labels.extend(labels)
            logging.info(f"Loaded {len(labels)} items from {batch}")
        except Exception as e:
            logging.warning(f"Failed loading {batch}: {e}")

    if all_embeddings:
        all_embeddings = np.vstack(all_embeddings)

    return all_embeddings, all_labels

import pandas as pd

def save_results_as_dataframe(results: dict, save_dir: str, experiment_type: str, batch: str):
    """
    Save evaluation results to a CSV file as a DataFrame with model names as index.

    Args:
        results (dict): Dictionary containing metrics for each model.
        save_dir (str): Directory where the CSV file should be saved.
        experiment_type (str): Name of the experiment.
        batch (str): Identifier for the batch or concatenated batches.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, experiment_type, batch, f"metrics.csv")

    # Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')

    # Save to CSV
    df.to_csv(path)
    logging.info(f"Saved metrics to {path}")


def save_pdist_matrix(pdist_values: np.ndarray, save_dir: str, model_name: str, experiment_type: str, batch: str):
    """
    Save the pairwise distance matrix to a file.
    Args:
        pdist_values (np.ndarray): Pairwise distance values.
        save_dir (str): Directory to save the pairwise distance matrix.
        model_name (str): Name of the model.
        experiment_type (str): Type of the experiment.
        batch (str): Identifier for the batch or concatenated batches.
    """
    logging.info(f"Saving pairwise distance matrix for {model_name} {experiment_type} {batch}")
    
    out_dir = os.path.join(save_dir, experiment_type, batch, "pdist", model_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"dists.npy")
    np.save(path, pdist_values)
    logging.info(f"Saved pdist to {path}")

def save_plot_alignment_vs_uniformity(results: dict, experiment_type:str, batch:str, filepath: str):
    """
    Save a scatter plot of alignment vs uniformity.
    Args:
        results (dict): Dictionary containing alignment and uniformity values for each model.
        filepath (str): Path to save the plot.
    """
    logging.info(f"Saving alignment vs uniformity plot to {filepath}")

    plt.figure(figsize=(8, 6))
    for model_name, (align, uniform) in results.items():
        plt.scatter(-uniform, -align, label=model_name, s=100)
        plt.text(-uniform, -align, model_name, fontsize=9)
    plt.xlabel('-Uniformity')
    plt.ylabel('-Alignment')
    plt.title(f'Embedding Comparison ({experiment_type} {batch})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    logging.info(f"Saved alignment vs uniformity plot to {filepath}")

def save_plot_alignment_vs_recall(results: dict, experiment_type:str, batch:str, filepath: str, k: int):
    """
    Save a scatter plot of alignment vs recall@k.
    Args:
        results (dict): Dictionary containing alignment and recall values for each model.
        filepath (str): Path to save the plot.
        k (int): Number of neighbors for recall@k.
    """

    logging.info(f"Saving alignment vs recall@{k} plot to {filepath}")
    
    plt.figure(figsize=(8, 6))
    for model_name, (align, recall) in results.items():
        plt.scatter(recall, -align, label=model_name, s=100)
        plt.text(recall, -align, model_name, fontsize=9)
    plt.xlabel(f'Recall@{k}')
    plt.ylabel('-Alignment')
    plt.title(f'-Alignment vs Recall@{k} ({experiment_type} {batch})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    logging.info(f"Saved alignment vs recall@{k} plot to {filepath}")

def save_plot_alignment_vs_entropy(results: dict, experiment_type:str, batch:str, filepath: str, k: int):
    """
    Save a scatter plot of alignment vs negative neighbor entropy.
    Args:
        results (dict): Dictionary containing alignment and entropy values for each model.
        filepath (str): Path to save the plot.
        k (int): Number of neighbors for negative neighbor entropy.
    """
    logging.info(f"Saving alignment vs entropy@{k} plot to {filepath}")

    plt.figure(figsize=(8, 6))
    for model_name, (align, entropy) in results.items():
        plt.scatter(-entropy, -align, label=model_name, s=100)
        plt.text(-entropy, -align, model_name, fontsize=9)
    plt.xlabel(f'-Entropy@{k}')
    plt.ylabel('-Alignment')
    plt.title(f'-Alignment vs Entropy@{k} ({experiment_type} {batch})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    logging.info(f"Saved alignment vs entropy@{k} plot to {filepath}")

def filter_out_dapi(embeddings: np.ndarray, labels: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Remove entries where the label starts with 'DAPI_'.

    Args:
        embeddings (np.ndarray): Embedding vectors of shape (N, D).
        labels (List[str]): Corresponding labels of length N.

    Returns:
        Tuple[np.ndarray, List[str]]: Filtered embeddings and labels.
    """
    assert len(embeddings) == len(labels), "Mismatch between embeddings and labels"

    # Identify indices to keep (not starting with 'DAPI_')
    keep_indices = [i for i, label in enumerate(labels) if not label.startswith("DAPI_")]

    # Filter
    filtered_embeddings = embeddings[keep_indices]
    filtered_labels = [labels[i] for i in keep_indices]

    logging.info(f"Filtered out {len(labels) - len(filtered_labels)} DAPI samples")

    return filtered_embeddings, filtered_labels

from collections import defaultdict
import random

def sample_by_label_fraction(
    embeddings: np.ndarray,
    labels: List[str],
    fraction: float,
    seed: int = 1
) -> Tuple[np.ndarray, List[str]]:
    """
    Sample a given fraction of data from each label group.

    Args:
        embeddings (np.ndarray): Array of shape (N, D) with all embeddings.
        labels (List[str]): List of N labels corresponding to the embeddings.
        fraction (float): Value between 0 and 1 indicating the proportion of each label group to keep.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, List[str]]: Subsampled embeddings and corresponding labels.
    """
    assert 0 < fraction <= 1.0, "fraction must be in (0, 1]"
    assert len(embeddings) == len(labels), "Mismatch between embeddings and labels"

    logging.info(f"Sampling {fraction*100:.1f}% of each label group")

    # Organize indices by label
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    random.seed(seed)
    selected_indices = []

    for label, idx_list in label_to_indices.items():
        k = max(1, int(len(idx_list) * fraction))
        sampled = random.sample(idx_list, k)
        selected_indices.extend(sampled)

    selected_indices.sort()
    sampled_embeddings = embeddings[selected_indices]
    sampled_labels = [labels[i] for i in selected_indices]

    logging.info(f"Selected {len(selected_indices)} out of {len(labels)} total examples")

    return sampled_embeddings, sampled_labels



# --- Run the full evaluation workflow for selected experiment/batch ---
def run_evaluation(model_folders_dict: Dict[str, str], experiment_type: str,
                    batches: List[str], pos_keys: List[str]=['marker', 'line', 'condition'],
                    precomputed_dists_paths: Dict[str, str] = None, sample_fraction:float = 1.0,
                    save_dir: str = "results", k: int = 5, neg_k: int = 10) -> None:
    """
    Run evaluation pipeline for a specific experiment and batches.

    Args:
        model_folders_dict (Dict[str, str]): Mapping of model names to their directory paths.
        experiment_type (str): Name of the experiment (e.g., 'neurons').
        batches (List[str]): List of batch names to process.
        pos_keys (List[str]): Keys to determine positive relationships in labels.
        precomputed_dists_paths (Dict[str, str]): Paths to precomputed distance matrices for each model.
        sample_fraction (float): Fraction of data to sample from each label group.
        save_dir (str): Directory to save metrics and plots.
        k (int): Number of neighbors for recall@k
        neg_k (int): Number of neighbors for negative neighbor entropy.

    Returns:
        None
    """
    results = {}

    for model_name, model_dir in model_folders_dict.items():
        logging.info(f"Evaluating model: {model_name}")

        embeddings, labels = load_model_data(model_dir, experiment_type, batches)

        # REMOVING DAPI
        logging.warning(f"NOTE: Removing DAPI from labels!!!")
        embeddings, labels = filter_out_dapi(embeddings, labels)
        
        # SAMPLING BY LABEL FRACTION
        logging.warning(f"NOTE: Sampling {experiment_type} by label fraction!!!")
        embeddings, labels = sample_by_label_fraction(embeddings, labels, fraction=sample_fraction)

        parsed_labels = [parse_label(l) for l in labels]
        formatted_labels = ['_'.join([d[k] for k in pos_keys if d[k]]) for d in parsed_labels]

        logging.info(f"[{model_name}] embeddings shape: {embeddings.shape}, labels: {len(formatted_labels)}")
        logging.info(f"[{model_name}] Unique formatted labels ({len(set(formatted_labels))}) : {set(formatted_labels)}")

        # Calculate pairwise distances
        precomputed_dists = precomputed_dists_paths.get(model_name, None) if precomputed_dists_paths else None
        if precomputed_dists is None:
            logging.info(f"[{model_name}] Calculating pairwise distances...")
        else:
            logging.info(f"[{model_name}] Loading precomputed pairwise distances from {precomputed_dists}")
            precomputed_dists = np.load(precomputed_dists)

        dists, pos_dists, neg_dists, pos_mask = compute_pairwise_distances_and_masks(embeddings, formatted_labels, pos_keys=pos_keys, dists=precomputed_dists)
        
        if precomputed_dists is None:
            # Save pairwise distance matrix
            save_pdist_matrix(dists, save_dir, model_name, experiment_type, '_'.join(batches))

        # Calculate metrics
        logging.info(f"[{model_name}] Calculating metrics...")
        logging.info(f"[{model_name}] Calculating alignment")
        alignment = compute_alignment(pos_dists, formatted_labels)
        logging.info(f"[{model_name}] Calculating uniformity")
        uniformity = compute_uniformity(neg_dists, formatted_labels)
        logging.info(f"[{model_name}] Calculating recall@{k}")
        recall = compute_recall_at_k(dists, formatted_labels, k=k)
        logging.info(f"[{model_name}] Calculating negative neighbor entropy@{neg_k}")
        entropy = compute_negative_neighbor_entropy(dists, pos_mask, formatted_labels, k=neg_k)

        results[model_name] = {
            "alignment": alignment,
            "uniformity": uniformity,
            "recall_at_k": recall,
            "entropy_at_k": entropy,
            "k": k,
            "neg_k": neg_k,
            "pos_keys": pos_keys
        }

    logging.info(f"[{model_name}] Saving results...")
    save_results_as_dataframe(results, save_dir, experiment_type, '_'.join(batches))

    return results

def save_plots(results: dict, save_dir: str, experiment_type: str, batches: List[str], k: int = 5, neg_k: int = 10)-> None:
    """
    Save various plots based on the evaluation results.

    Args:
        results (dict): Dictionary containing metrics for each model.
        save_dir (str): Directory to save the plots.
        experiment_type (str): Name of the experiment.
        batches (List[str]): List of batch names.
        k (int): Number of neighbors for recall@k.
        neg_k (int): Number of neighbors for negative neighbor entropy.

    Returns:
        None
    """

    # Save plots
    batches = '_'.join(batches)
    plot_dir = os.path.join(save_dir, experiment_type, batches, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    results_align_uniformity = {m: (r['alignment'], r['uniformity']) for m, r in results.items()}
    results_align_recall = {m: (r['alignment'], r['recall_at_k']) for m, r in results.items()}
    results_align_entropy = {m: (r['alignment'], r['entropy_at_k']) for m, r in results.items()}
    save_plot_alignment_vs_uniformity(results_align_uniformity, experiment_type, batches, os.path.join(plot_dir, f"align_uniform.png"))
    save_plot_alignment_vs_recall(results_align_recall, experiment_type, batches, os.path.join(plot_dir, f"align_recall.png"), k)
    save_plot_alignment_vs_entropy(results_align_entropy, experiment_type, batches, os.path.join(plot_dir, f"align_entropy.png"), k=neg_k)

def init_logging(path):
    """Init logging.
    Writes to log file and console.
    Args:
        path (string): Path to log file
    """
  
    jobid = os.getenv('LSB_JOBID')
    jobname = os.getenv('LSB_JOBNAME')
    # if jobname is not specified, the jobname will include the path of the script that was run.
    # In this case we'll have some '/' and '.' in the jobname that should be removed.
    if jobname:
        jobname = jobname.replace('/','').replace('.','') 

    username = 'UnknownUser'
    if jobid:
        # Run the bjobs command to get job details
        result = subprocess.run(['bjobs', '-o', 'user', jobid], capture_output=True, text=True, check=True)
        # Extract the username from the output
        username = result.stdout.replace('USER', '').strip()
    
    __now_str = datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f")
    log_file_path = os.path.join(path, __now_str + f'_{jobid}_{username}_{jobname}.log')
    if not os.path.exists(path):
        os.makedirs(path)
        
    logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler()
                    ])

    logging.info(f"Init (log path: {log_file_path}; JOBID: {jobid} Username: {username}) JOBNAME: {jobname}")
    logging.info(f"NOVA_HOME={os.getenv('NOVA_HOME')}, NOVA_DATA_HOME={os.getenv('NOVA_DATA_HOME')}")

def parse_positional_args(argv):
    """
    Parse positional arguments in the following order:
    1. experiment (str)
    2. batch (str)
    3. k (int)
    4. neg_k (int)
    5. save_dir (str, optional)

    Args:
        argv (List[str]): List of command-line arguments.

    Returns:
        dict: Parsed values.
    """
    if len(argv) < 4 or len(argv) > 7:
        raise ValueError("Usage: script.py <experiment> <batch> <save_dir> [k] [neg_k] [sample_fraction]")

    experiment = argv[1]
    batch = argv[2]
    save_dir = argv[3]
    k = int(argv[4]) if len(argv) > 3 else 20
    neg_k = int(argv[5]) if len(argv) > 4 else 20
    sample_fraction = float(argv[6]) if len(argv) > 5 else 1.0

    return {
        'experiment': experiment,
        'batches': [batch],  # Wrapped in list for compatibility with rest of pipeline
        'save_dir': save_dir,
        'k': k,
        'neg_k': neg_k,
        'sample_fraction': sample_fraction
    }


if __name__ == "__main__":
    args = parse_positional_args(sys.argv)

    experiment = args['experiment']
    batches = args['batches']
    save_dir = args['save_dir']
    k = args['k']
    neg_k = args['neg_k']
    sample_fraction = args['sample_fraction']

    model_folders_dict = {
            'pretrained': '/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/outputs/vit_models_local/pretrained_model',  
            'finetuned_CL': '/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/outputs/vit_models_local/finetuned_model',  
            'finetuned_CL_nofreeze': '/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/outputs/vit_models_local/finetuned_model_no_freeze',  
            'finetuned_CE_nofreeze': '/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/outputs/vit_models_local/finetuned_model_classification_with_batch_no_freeze',
            'finetuned_CE': "/home/projects/hornsteinlab/Collaboration/MOmaps_Sagy/NOVA/outputs/vit_models_local/finetuned_model_classification_with_batch_freeze"
        }

    precomputed_dists_paths = {
            'pretrained': None, 
            'finetuned_CL': None,
            'finetuned_CL_nofreeze': None, 
            'finetuned_CE_nofreeze': None,
            'finetuned_CE': None
        }

    init_logging(os.path.join(save_dir, experiment, '_'.join(batches), 'logs'))
    logging.info(f"Model folders: {model_folders_dict}; Experiment: {experiment}; Batches: {batches}; k: {k}; neg_k: {neg_k} ; Save dir: {save_dir}; Sample fraction: {sample_fraction}")
    
    try:
        results = run_evaluation(model_folders_dict, experiment_type=experiment, batches=batches, precomputed_dists_paths=precomputed_dists_paths, save_dir=save_dir, k=k, neg_k=neg_k, sample_fraction=sample_fraction)
        save_plots(results, save_dir=save_dir, experiment_type=experiment, batches=batches, k=k, neg_k=neg_k)
    except Exception as e:
        logging.exception(f"Error during evaluation {str(e)}")
        raise
