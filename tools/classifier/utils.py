import pickle
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    confusion_matrix,
    roc_curve, auc
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.decomposition import PCA


from src.analysis.analyzer_classification_utils import (
    run_baseline_model,
    run_train_test_split_baseline, load_batches, concat_from_cache,
    remove_untreated_from_labels, plot_confusion_matrix
)
    
def load_batches_pkl(batch_ids, umap=1):
    """Load and concatenate embeddings and labels from pkl files based on a list of batch numbers."""
    X_list, y_list = [], []
    for b in batch_ids:
        with open(f"data/newNeuronsD8FigureConfig_UMAP{umap}_B{b}.pkl", "rb") as f:
            data = pickle.load(f)
            X_list.append(data["embeddings"])
            y_list.append(data["labels"])
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

def plot_multiclass_roc(y_true, y_scores, classes, title="ROC (OvR)"):
    classes = np.asarray(classes)

    # map ints -> names if needed
    y_true = np.asarray(y_true)
    if np.issubdtype(y_true.dtype, np.integer):
        y_true = classes[y_true]

    # binarize in class order
    lb = LabelBinarizer().fit(classes)
    Y = lb.transform(y_true)

    # handle binary shapes from LabelBinarizer:
    # it returns (n_samples,) OR (n_samples, 1) for binary problems.
    if Y.ndim == 1:
        Y = np.column_stack([1 - Y, Y])
    elif Y.shape[1] == 1 and len(classes) == 2:
        Y = np.column_stack([1 - Y[:, 0], Y[:, 0]])

    # (optional) quick sanity: columns must match
    if y_scores.shape[1] != len(classes):
        raise ValueError(f"y_scores has {y_scores.shape[1]} cols, classes={len(classes)}")

    # skip classes with no positives
    fpr, tpr, roc_auc, valid = {}, {}, {}, []
    for i, cls in enumerate(classes):
        if Y[:, i].sum() == 0:
            continue
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        valid.append(i)

    # micro/macro (optional)
    if valid:
        plt.figure()
        fpr_micro, tpr_micro, _ = roc_curve(Y[:, valid].ravel(), y_scores[:, valid].ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        plt.plot(fpr_micro, tpr_micro, "--", label=f"micro (AUC={auc_micro:.2f})")

        all_fpr = np.unique(np.concatenate([fpr[i] for i in valid]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in valid:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(valid)
        auc_macro = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, "--", label=f"macro (AUC={auc_macro:.2f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(title + " (micro/macro)")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout(); plt.show()

    # per-class
    plt.figure()
    for i in valid:
        cls = classes[i]
        plt.plot(fpr[i], tpr[i], label=f"{cls} (AUC={roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.show()

def spearman_gpu(X, y, top_n=100):    
    import cupy as cp

    # Convert to GPU
    X_gpu = cp.asarray(X)
    y_gpu = cp.asarray(y)

    # Rank columns of X and y
    X_ranked = X_gpu.argsort(axis=0).argsort(axis=0).astype(cp.float32)
    y_ranked = y_gpu.argsort().argsort().astype(cp.float32)

    # Centered ranks
    X_ranked -= X_ranked.mean(axis=0)
    y_ranked -= y_ranked.mean()

    # Compute numerator and denominator of correlation
    numerator = cp.sum(X_ranked * y_ranked[:, None], axis=0)
    denominator = cp.sqrt(cp.sum(X_ranked**2, axis=0) * cp.sum(y_ranked**2))
    correlations = numerator / denominator

    # Get top N features by absolute correlation
    top_indices = cp.argsort(cp.abs(correlations))[-top_n:][::-1].get()
    return top_indices, correlations[top_indices].get()

def run_cluster_eval(dataset_config, batches, pca_options, norm_options, balance_options, cluster_range, method = 'kmeans'):
    results = []

    for pca_flag, norm_flag, balance_flag, n_clusters in product(pca_options, norm_options, balance_options, cluster_range):
        print(f"Running: PCA={pca_flag}, Normalize={norm_flag}, Balance={balance_flag}, Clusters={n_clusters}")
        cluster_kwargs = {'n_clusters' if method == 'kmeans' else 'n_components': n_clusters} if method in ['kmeans', 'gmm'] else {}

        df, _, _ = run_clustering(dataset_config=dataset_config, batches=batches, method=method, cluster_kwargs=cluster_kwargs,
                                  normalize=norm_flag, balance=balance_flag, apply_pca=pca_flag)

        # Cluster purity: each cluster dominated by 1 label
        cluster_purity = (
            df.groupby('cluster')['label'].value_counts(normalize=True)
            .groupby(level=0).max().mean()
        )

        # Label consistency: each label mostly in one cluster
        label_consistency = (
            df.groupby('label')['cluster'].value_counts(normalize=True)
            .groupby(level=0).max().mean()
        )

        results.append({
            'pca': pca_flag,
            'normalize': norm_flag,
            'balance': balance_flag,
            'n_clusters': n_clusters,
            'cluster_purity': cluster_purity,
            'label_consistency': label_consistency
        })

    return pd.DataFrame(results)

def run_clustering(
    dataset_config,
    batches,
    method='kmeans',  # 'kmeans', 'gmm', or 'dbscan'
    normalize=False,
    balance=False,
    apply_pca=False,
    pca_components=50,
    cluster_kwargs=dict()  # e.g., {'n_clusters': 5} or {'eps': 1.2}
):
    # Load data via cache, then concat the requested batches
    cache = load_batches(batches, dataset_config)
    X_all, y_all = concat_from_cache(cache, batches)
    print('Data loaded.')
    y_all = np.array(y_all)

    # Optional: Balance dataset
    if balance: 
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        X_all, y_all = ros.fit_resample(X_all, y_all)

    # Optional: Normalize
    if normalize:
        scaler = StandardScaler()
        X_all = scaler.fit_transform(X_all)

    # Optional: PCA
    if apply_pca:
        X_all = PCA(n_components=pca_components, random_state=42).fit_transform(X_all)

    # Clustering
    method = method.lower()
    if method == 'kmeans':
        model = KMeans(random_state=42, **cluster_kwargs)
    elif method == 'gmm':
        model = GaussianMixture(random_state=42, **cluster_kwargs)
    elif method == 'dbscan':
        model = DBSCAN(**cluster_kwargs)
    else:
        raise ValueError(f"Unsupported method: {method}")

    clusters = model.fit_predict(X_all)

    # Build DataFrame
    df = pd.DataFrame({'cluster': clusters, 'label': y_all})

    # Cluster → Label distribution
    cluster_summary = plot_cluster_label_distribution(df, row='cluster', col='label', cmap='Blues',
        title=f"Label % Distribution per {method.upper()} Cluster")

    # Label → Cluster distribution
    label_summary = plot_cluster_label_distribution(df, row='label', col='cluster', cmap='Purples',
        title=f"Cluster % Distribution per Label ({method.upper()})")

    # Merge clusters by label composition
    df_with_merged, cluster_map = unify_clusters_by_label_composition(df, cluster_summary, label_threshold=5.0)

    # Merged cluster → Label distribution
    merged_summary = plot_cluster_label_distribution(df_with_merged, row='merged_cluster', col='label', cmap='Greens',
        title="Label % Distribution per Merged Cluster")
    
    # Merged label → Cluster distribution
    merged_label_summary = plot_cluster_label_distribution(df_with_merged, row='label', col='merged_cluster', cmap='Oranges',
        title="Merged Cluster % Distribution per Label")
    
    create_and_plot_cluster_confusion_matrix(df_with_merged, cluster_col='merged_cluster', title_prefix=f"{method.upper()} ")

    return df, cluster_summary, label_summary

def create_and_plot_cluster_confusion_matrix(df, cluster_col='cluster', title_prefix=''):
    """
    Plots a confusion matrix between true labels and predicted clusters.

    Parameters:
        df (pd.DataFrame): Must contain 'label' and 'cluster' or 'merged_cluster' columns
        cluster_col (str): 'cluster' or 'merged_cluster'
        title_prefix (str): Optional prefix for the plot title

    Returns:
        cm (np.ndarray): Confusion matrix array
    """
    labels = sorted(df['label'].unique())
    clusters = sorted(df[cluster_col].unique())

    cm = confusion_matrix(df['label'], df[cluster_col], labels=labels)

    plot_confusion_matrix(cm, clusters, title=f"{title_prefix}Confusion Matrix: True Label vs {cluster_col}",
                          xlabel="Cluster", ylabel="True Label")

    return cm

def plot_cluster_label_distribution(df, row='cluster', col='label', cmap='Blues', title=''):
    summary = (
        df.groupby(row)[col].value_counts(normalize=True)
        .unstack()
        .fillna(0) * 100
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(summary, annot=True, fmt=".1f", cmap=cmap)
    plt.title(title)
    plt.xlabel(col.capitalize())
    plt.ylabel(row.capitalize())
    plt.tight_layout()
    plt.show()
    return summary

def unify_clusters_by_label_composition(df, cluster_summary, label_threshold=2.0):
    """
    Groups clusters based on similar label composition.
    A label is considered 'present' in a cluster if it occupies more than `label_threshold` percent.

    Parameters:
        df: pd.DataFrame with 'cluster' and 'label' columns
        cluster_summary: DataFrame, cluster x label percentage
        label_threshold: float, percent threshold to consider label as present

    Returns:
        df with added column 'merged_cluster'
        cluster_groups: dict mapping original clusters to merged cluster IDs
    """
    # Step 1: define a set of "present" labels per cluster
    def get_present_labels(row):
        return frozenset(label for label, pct in row.items() if pct > label_threshold)

    label_sets = cluster_summary.apply(get_present_labels, axis=1)

    # Step 2: assign each unique label set to a group ID
    unique_sets = {}
    cluster_to_group = {}
    group_counter = 0
    for cluster_id, label_set in label_sets.items():
        if label_set not in unique_sets:
            unique_sets[label_set] = group_counter
            group_counter += 1
        cluster_to_group[cluster_id] = unique_sets[label_set]

    # Step 3: apply mapping to the original df
    df['merged_cluster'] = df['cluster'].map(cluster_to_group)

    return df, cluster_to_group
