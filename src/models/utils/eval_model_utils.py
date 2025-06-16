import os
import sys
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import pairwise_distances

sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.models.utils.eval_model_metrics import compute_alignment, compute_negative_neighbor_entropy, compute_precision_at_k, compute_recall, compute_uniformity, linear_probe_cv
from src.embeddings.embeddings_config import EmbeddingsConfig
from src.embeddings.embeddings_utils import load_embeddings

def run_evaluation(model_folders_dict: Dict[str, str], embeddings_config: EmbeddingsConfig,
                    precomputed_dists_paths: Dict[str, str] = None, sample_fraction:float = 1.0,
                    save_dir: str = "results", k: int = 5, neg_k: int = 10) -> None:
    """
    Run evaluation pipeline for a specific experiment and batches.

    Args:
        model_folders_dict (Dict[str, str]): Mapping of model names to their directory paths.
        embeddings_config (EmbeddingsConfig): Configuration for loading embeddings.
        precomputed_dists_paths (Dict[str, str]): Paths to precomputed distance matrices for each model.
        sample_fraction (float): Fraction of data to sample from each label group.
        save_dir (str): Directory to save metrics and plots.
        k (int): Number of neighbors for precision
        neg_k (int): Number of neighbors for negative neighbor entropy.

    """
    results = {}

    experiment_type = embeddings_config.EXPERIMENT_TYPE
    batches = "_".join([folder.split(os.sep)[-1] for folder in embeddings_config.INPUT_FOLDERS])

    for model_name, model_dir in model_folders_dict.items():
        logging.info(f"Evaluating model: {model_name}")

        embeddings, labels, _ = load_embeddings(model_dir, embeddings_config, sample_fraction=sample_fraction)
        logging.info(f"Embeddings shape: {embeddings.shape}")

        logging.info(pd.value_counts(labels))

        logging.info(f"[{model_name}] embeddings shape: {embeddings.shape}, labels: {len(labels)}")

        # Calculate pairwise distances
        precomputed_dists_path = precomputed_dists_paths.get(model_name, None) if precomputed_dists_paths is not None else None
        dists, pos_dists, neg_dists, pos_mask = __compute_pairwise_distances_and_masks(embeddings, labels, precomputed_dists_path=precomputed_dists_path)
        
        if precomputed_dists_path is None:
            __save_pairwise_distances(dists, save_dir, model_name, experiment_type, batches)

        results[model_name] = __calculate_metrics(model_name, dists, pos_dists, neg_dists, pos_mask,
                                                  embeddings, labels, k=k, neg_k=neg_k, seed=embeddings_config.SEED)

    logging.info(f"[{model_name}] Saving results...")
    __save_results_as_dataframe(results, save_dir, experiment_type, batches)

    return results

def save_plots(results: dict, save_dir: str, experiment_type: str, batches: List[str], k: int = 5, neg_k: int = 10, postfix: str = "")-> None:
    """
    Save various plots based on the evaluation results.

    Args:
        results (dict): Dictionary containing metrics for each model.
        save_dir (str): Directory to save the plots.
        experiment_type (str): Name of the experiment.
        batches (List[str]): List of batch names.
        k (int): Number of neighbors for recall@k.
        neg_k (int): Number of neighbors for negative neighbor entropy.
        postfix (str): Postfix to append to saved plot filenames.

    Returns:
        None
    """
    plot_dir = os.path.join(save_dir, experiment_type, batches, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    __save_plot_alignment_vs_uniformity(results, experiment_type, batches, os.path.join(plot_dir, f"nAlignment_nUniformity{postfix}.png"))
    __save_plot_alignment_vs_recall(results, experiment_type, batches, os.path.join(plot_dir, f"nAlignment_Recall{postfix}.png"))
    __save_plot_alignment_vs_entropy(results, experiment_type, batches, os.path.join(plot_dir, f"nAlignment_nEntropy{neg_k}{postfix}.png"), k=neg_k)
    __save_plot_alignment_vs_accuracy(results, experiment_type, batches, os.path.join(plot_dir, f"nAlignment_Accuracy{postfix}.png"))
    __save_plot_alignment_vs_precision(results, experiment_type, batches, os.path.join(plot_dir, f"nAlignment_Precision{k}{postfix}.png"), k)
    __save_plot_precision_vs_recall(results, experiment_type, batches, os.path.join(plot_dir, f"nPrecision_Recall{postfix}.png"), k)


def aggregate_and_plot_metrics(save_dir: str, postfix: str = "_average") -> None:
    """
    Aggregate all metrics.csv files from subdirectories of save_dir,
    compute the average metrics per model, and save the results and plots.

    Args:
        save_dir (str): Directory containing subdirectories with *_metrics.csv files.
        postfix (str): Suffix to append to saved plot filenames.

    """
    metrics_files = []
    for root, _, files in os.walk(save_dir):
        for file in files:
            if file == "metrics.csv":
                metrics_files.append(os.path.join(root, file))

    if not metrics_files:
        raise FileNotFoundError(f"No metrics CSV files found in {save_dir}. Please ensure that the evaluation has been run and metrics files are generated.")

    experiments_df = []
    for path in metrics_files:
        df = pd.read_csv(path, index_col=0)
        experiments_df.append(df)

    combined_df = pd.concat(experiments_df)
    avg_df = combined_df.groupby(combined_df.index).mean()

    # Save averaged CSV
    avg_csv_path = os.path.join(save_dir, f"metrics{postfix}.csv")
    avg_df.to_csv(avg_csv_path)

    # Save the list of used files
    used_list_path = os.path.join(save_dir, f"sources{postfix}.txt")
    with open(used_list_path, 'w') as f:
        for path in metrics_files:
            f.write(path + '\n')

    results = avg_df[['alignment', 'uniformity', 'precision_at_k', 'recall', 'entropy_at_k', 'accuracy']].to_dict(orient='index')

    save_plots(results, save_dir, "Average", "", k=int(avg_df['k'].iloc[0]), neg_k=int(avg_df['neg_k'].iloc[0]), postfix=postfix)

    return avg_df

def __compute_pairwise_distances_and_masks(embeddings: np.ndarray, labels: List[str], precomputed_dists_path:str=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pairwise distances and extract the positive and negative distance vectors based on label similarity.

    Args:
        embeddings (np.ndarray): Embedding vectors.
        labels (List[str]): Corresponding labels.
        precomputed_dists (str, optional): Path to precomputed pairwise distances. If None, distances will be computed.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: All distances, positive distances, negative distances, positive mask.
    """

    if precomputed_dists_path is None:
        logging.info(f"Calculating pairwise distances...")
        dists = pairwise_distances(embeddings, metric='euclidean', n_jobs=-1)
    else:
        logging.info(f"Loading precomputed pairwise distances from {precomputed_dists_path}")
        dists = np.load(precomputed_dists_path)

    # Create a boolean mask for positive pairs based on the specified keys
    pos_mask = np.equal.outer(labels, labels)

    # Ensure the diagonal is False (no self-comparisons)
    np.fill_diagonal(pos_mask, False)

    pos_dists = dists[pos_mask]
    neg_dists = dists[~pos_mask]

    return dists, pos_dists, neg_dists, pos_mask

def __save_results_as_dataframe(results: dict, save_dir: str, experiment_type: str, batch: str):
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


def __save_pairwise_distances(pdist_values: np.ndarray, save_dir: str, model_name: str, experiment_type: str, batch: str):
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

def __save_plot(results: dict, metric1_key_name:str, metric2_key_name:str, metric1_display_name:str, metric2_display_name:str, experiment_type:str, batch:str, filepath: str):
    """
    Save a scatter plot of alignment vs uniformity.
    Args:
        results (dict): Dictionary containing alignment and uniformity values for each model.
        metrci1_key_name (str): Key for the first metric in results.
        metric2_key_name (str): Key for the second metric in results.
        metric1_display_name (str): Display name for the first metric.
        metric2_display_name (str): Display name for the second metric.
        experiment_type (str): Type of the experiment.
        batch (str): Identifier for the batch or concatenated batches.
        filepath (str): Path to save the plot.
    """
    logging.info(f"Saving alignment vs uniformity plot to {filepath}")

    plt.figure(figsize=(8, 6))
    for model_name, metrics in results.items():
        metric1_value = metrics[metric1_key_name]
        metric2_value = metrics[metric2_key_name]
        plt.scatter(metric1_value, metric2_value, label=model_name, s=200)
        plt.text(metric1_value, metric2_value, model_name, fontsize=12)
    plt.xlabel(metric1_display_name)
    plt.ylabel(metric2_display_name)
    plt.title(f'Embedding Comparison ({experiment_type} {batch})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    logging.info(f"Saved {metric1_display_name} vs {metric2_display_name} plot to {filepath}")

def __save_plot_alignment_vs_uniformity(results: dict, experiment_type:str, batch:str, filepath: str):
    """
    Save a scatter plot of alignment vs uniformity.
    Args:
        results (dict): Dictionary containing alignment and uniformity values for each model.
        filepath (str): Path to save the plot.
    """
    logging.info(f"Saving alignment vs uniformity plot to {filepath}")
    __save_plot(results, 'uniformity', 'alignment', '-Uniformity', '-Alignment', experiment_type, batch, filepath)

def __save_plot_alignment_vs_recall(results: dict, experiment_type:str, batch:str, filepath: str):
    """
    Save a scatter plot of alignment vs recall.
    Args:
        results (dict): Dictionary containing alignment and recall values for each model.
        filepath (str): Path to save the plot.
        k (int): Number of neighbors for recall.
    """

    logging.info(f"Saving alignment vs recall plot to {filepath}")
    __save_plot(results, 'recall', 'alignment', f'Recall', '-Alignment', experiment_type, batch, filepath)

def __save_plot_alignment_vs_precision(results: dict, experiment_type:str, batch:str, filepath: str, k: int):
    """
    Save a scatter plot of alignment vs precision@k.
    Args:
        results (dict): Dictionary containing alignment and recall values for each model.
        filepath (str): Path to save the plot.
        k (int): Number of neighbors for precision@k.
    """

    logging.info(f"Saving alignment vs precision@{k} plot to {filepath}")
    __save_plot(results, 'precision_at_k', 'alignment', f'Precision@{k}', '-Alignment', experiment_type, batch, filepath)

def __save_plot_precision_vs_recall(results: dict, experiment_type:str, batch:str, filepath: str, k: int):
    """
    Save a scatter plot of precision@k vs recall
    Args:
        results (dict): Dictionary containing alignment and recall values for each model.
        filepath (str): Path to save the plot.
        k (int): Number of neighbors for precision@k.
    """

    logging.info(f"Saving recall vs precision@{k} plot to {filepath}")
    __save_plot(results, 'recall', 'precision_at_k', f'Recall', f'Precision@{k}', experiment_type, batch, filepath)
    

def __save_plot_alignment_vs_entropy(results: dict, experiment_type:str, batch:str, filepath: str, k: int):
    """
    Save a scatter plot of alignment vs negative neighbor entropy.
    Args:
        results (dict): Dictionary containing alignment and entropy values for each model.
        filepath (str): Path to save the plot.
        k (int): Number of neighbors for negative neighbor entropy.
    """
    logging.info(f"Saving alignment vs entropy@{k} plot to {filepath}")

    __save_plot(results, 'entropy_at_k', 'alignment', f'-Negative Neighbor Entropy@{k}', '-Alignment', experiment_type, batch, filepath)

def __save_plot_alignment_vs_accuracy(results: dict, experiment_type:str, batch:str, filepath: str):
    """
    Save a scatter plot of alignment vs accuracy.
    Args:
        results (dict): Dictionary containing alignment and accuracy values for each model.
        experiment_type (str): Type of the experiment.
        filepath (str): Path to save the plot.
    """
    logging.info(f"Saving alignment vs accuracy plot to {filepath}")

    __save_plot(results, 'accuracy', 'alignment', f'Accuracy', '-Alignment', experiment_type, batch, filepath)


def __calculate_metrics(model_name:str, dists: np.ndarray, pos_dists: np.ndarray, neg_dists: np.ndarray, pos_mask: np.ndarray,
                       embeddings: np.ndarray, labels: List[str], k: int = 5, neg_k: int = 10, seed=1):
    """ Calculate and store evaluation metrics for a model.

    Args:
        model_name (str): Name of the model being evaluated.
        dists (np.ndarray): Pairwise distance matrix of shape (N, N).
        pos_dists (np.ndarray): Positive pairwise distances of shape (P,).
        neg_dists (np.ndarray): Negative pairwise distances of shape (N-P,).
        pos_mask (np.ndarray): Boolean mask indicating positive pairs of shape (N, N).
        labels (List[str]): List of labels corresponding to the embeddings.
        k (int, optional): K positives. Defaults to 5.
        neg_k (int, optional): K negatives. Defaults to 10.
        seed (int, optional): Random seed for reproducibility. Defaults to 1.
    """
    
    # Calculate metrics
    logging.info(f"[{model_name}] Calculating metrics...")

    logging.info(f"[{model_name}] Calculating alignment")
    alignment = compute_alignment(pos_dists)

    logging.info(f"[{model_name}] Calculating uniformity")
    uniformity = compute_uniformity(neg_dists)

    logging.info(f"[{model_name}] Calculating recall")
    recall = compute_recall(dists, labels)

    logging.info(f"[{model_name}] Calculating precision@{k}")
    precision = compute_precision_at_k(dists, labels, k=k)

    logging.info(f"[{model_name}] Calculating negative neighbor entropy@{neg_k}")
    entropy = compute_negative_neighbor_entropy(dists, pos_mask, labels, k=neg_k)
    
    logging.info(f"[{model_name}] Calculating accuracy")
    accuracy = linear_probe_cv(embeddings, labels, kfold=3, seed=seed)

    return {
        "alignment": alignment,
        "uniformity": uniformity,
        "precision_at_k": precision,
        "recall": recall,
        "entropy_at_k": entropy,
        "accuracy": accuracy,
        "k": k,
        "neg_k": neg_k
    }


