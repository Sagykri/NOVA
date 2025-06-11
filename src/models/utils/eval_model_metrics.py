from collections import Counter
from typing import List, Tuple
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def compute_alignment(pos_dists:np.ndarray, alpha: float = 2) -> float:
    """ Compute alignment and uniformity metrics based on positive and negative distances.
    Higher values are better, indicating that the model is aligning positive distances.
    Args:
        pos_dists (np.ndarray): Array of positive distances.
        labels (List[str]): List of label strings.
        alpha (float): Exponent for alignment calculation.
    Returns:
        float: Alignment metric.
    """

    # Calculate alignment on positives
    alignment = np.mean(pos_dists ** alpha) if len(pos_dists) > 0 else np.nan

    alignment = -alignment

    return alignment

def compute_uniformity(neg_dists: np.ndarray, t: float = 2.0) -> float:
    """
    Compute uniformity metric based on negative distances.
    Higher values are better, indicating that the model is uniformly distributing negative distances.
    This metric is often used to evaluate the quality of embeddings in terms of how uniformly they spread out negative samples.
    Args:
        neg_dists (np.ndarray): Array of negative distances.
        labels (List[str]): List of label strings.
        t (float): Temperature parameter for uniformity calculation.
    Returns:
        float: Uniformity metric.
    """

    # Calculate uniformity on negatives
    uniformity = np.log(np.mean(np.exp(-t * neg_dists))) if len(neg_dists) > 0 else np.nan

    uniformity = -uniformity

    return uniformity

def compute_recall(distances: np.ndarray, labels: List[str]) -> float:
    labels = np.array(labels)

    distances_copy = distances.copy()
    # Ignore self-match
    np.fill_diagonal(distances_copy, np.inf)

    recalls = []
    
    for i, label in enumerate(labels):
        # Get k as the number of samples with the same label as current sample
        k = np.sum(labels == label) - 1  # Subtract 1 to exclude the sample itself
        
        if k == 0:  # If there are no other samples with the same label
            continue
            
        # Get indices of top-k nearest neighbors for current sample
        nearest = np.argsort(distances_copy[i])[:k]
        nearest_labels = labels[nearest]
        
        # Count matches between true label and neighbors
        matches = np.sum(nearest_labels == label)
        
        # Calculate recall for this sample
        recall_i = matches / k
        recalls.append(recall_i)
    
    # Return average recall across all samples
    return np.mean(recalls) if recalls else 0.0

def compute_precision_at_k(distances: np.ndarray, labels: List[str], k: int = 5) -> float:
    """
    Compute precision@k
    
    Args:
        distances (np.ndarray): Pairwise distance matrix of shape (N, N).
        labels (List[str]): List of N string labels.
        k (int): Number of top nearest neighbors to consider.

    Returns:
        precision@k (float)
    """
    labels = np.array(labels)

    distances_copy = distances.copy()
    # Ignore self-match
    np.fill_diagonal(distances_copy, np.inf)

    # Get indices of top-k nearest neighbors for each sample
    nearest = np.argsort(distances_copy, axis=1)[:, :k]
    nearest_labels = labels[nearest]               # (N, k)
    true_labels = labels[:, None]                  # (N, 1)

    # Count matches between true label and neighbors
    matches = (nearest_labels == true_labels)      # (N, k), bool
    retrieved_relevant = matches.sum(axis=1)       # (N,)

    # Calculate precision
    precision = (retrieved_relevant / k).mean()

    return precision

def compute_negative_neighbor_entropy(distances: np.ndarray, pos_mask: np.ndarray, labels: List[str], k: int = 5) -> float:
    """
    Compute the entropy of the top-k negative neighbors for each embedding.
    Higher values are better, indicating that the model has a diverse set of negative neighbors.
    

    Args:
        distances (np.ndarray): Pairwise distance matrix.
        pos_mask (np.ndarray): Boolean mask indicating positive neighbors.
        labels (List[str]): List of label strings.
        k (int): Number of top neighbors to consider.

    Returns:
        float: Average entropy of top-k negative neighbors across all embeddings.
    """
    entropies = []

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

    
    return -float(np.mean(entropies)) if entropies else 0.0



def linear_probe_cv(embeddings: np.ndarray, labels: np.ndarray, kfold:int=3, seed: int = 42) -> float:
    """
    Performs linear probing using logistic regression with k-fold stratified cross-validation.

    Args:
        embeddings (np.ndarray): Feature matrix of shape (N, D).
        labels (np.ndarray): Array of labels of shape (N,).
        kfold (int, Optional): Number of folds for cross-validation. Detault is 3.
        seed (int, Optional): Random seed for reproducibility. Default is 42.

    Returns:
        float: Mean accuracy across the k folds.
    """
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
    accuracies = []

    for train_idx, test_idx in skf.split(embeddings, labels):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)

    return float(np.mean(accuracies))
