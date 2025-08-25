import datetime as dt
import json
import os
import pickle
from collections import Counter, defaultdict
from itertools import product

import cupy as cp
import cudf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cuml.linear_model import LogisticRegression as cuMLLogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    multilabel_confusion_matrix,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import f_classif

from src.analysis.analyzer_multiplex_markers import AnalyzerMultiplexMarkers
from src.common.utils import load_config_file
from src.embeddings.embeddings_utils import load_embeddings

def count_labels(y):
    counts = Counter(y)
    for label, count in counts.items():
        print(f"{label}: {count}")
        
def plot_confusion_matrix(y_true, y_pred, label_encoder, shorten_labels=True, rotation=90):
    """
    Plots a confusion matrix with correctly aligned labels and optional label cleaning.
    """
    # Get original class labels
    labels = list(label_encoder.classes_)

    # Clean labels if needed
    if shorten_labels:
        display_labels = [l.replace('_Untreated', '') for l in labels]
    else:
        display_labels = labels

    # Compute confusion matrix using string labels
    y_true_str = label_encoder.inverse_transform(y_true)
    y_pred_str = label_encoder.inverse_transform(y_pred)
    cm = confusion_matrix(y_true_str, y_pred_str, labels=labels)

    # Plot    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(xticks_rotation=rotation, ax=ax, cmap='viridis') 
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # give extra space at the bottom
    plt.show()
    
def load_batches_pkl(batch_ids, umap=1):
    """Load and concatenate embeddings and labels from pkl files based on a list of batch numbers."""
    X_list, y_list = [], []
    for b in batch_ids:
        with open(f"data/newNeuronsD8FigureConfig_UMAP{umap}_B{b}.pkl", "rb") as f:
            data = pickle.load(f)
            X_list.append(data["embeddings"])
            y_list.append(data["labels"])
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

def load_batches(
    batch_ids,
    dataset_config 
):
    """
    Load and concatenate embeddings and labels across batches.

    Requires: load_config_file, load_embeddings, AnalyzerMultiplexMarkers (if multiplexed=True)
    """
    path_to_embeddings = dataset_config['path_to_embeddings']
    multiplexed=dataset_config.get('multiplexed', False)
    config_fmt=dataset_config.get('config_fmt', 'newNeuronsD8FigureConfig_UMAP1_B{batch}')
    config_dir=dataset_config.get('config_dir','manuscript/manuscript_figures_data_config')

    X_list, y_list = [], []

    for b in batch_ids:
        config_name = config_fmt.format(batch=b)
        config_path_data = f'{config_dir}/{config_name}'
        config_data = load_config_file(config_path_data, 'data')
        config_data.OUTPUTS_FOLDER = path_to_embeddings

        embeddings, labels, _ = load_embeddings(path_to_embeddings, config_data)

        if multiplexed:
            analyzer = AnalyzerMultiplexMarkers(config_data, path_to_embeddings)
            embeddings, labels, _ = analyzer.calculate(embeddings, labels)

        X_list.append(np.asarray(embeddings))
        y_list.append(np.asarray(labels))

    if not X_list:
        raise ValueError("No batches loaded.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0).reshape(-1)
    return X, y

def load_all_batches(batch_ids, dataset_config):
    """
    Return a dict: {batch_id: (X, y)}
    """
    path_to_embeddings = dataset_config['path_to_embeddings']
    multiplexed = dataset_config.get('multiplexed', False)
    config_fmt   = dataset_config.get('config_fmt', 'newNeuronsD8FigureConfig_UMAP1_B{batch}')
    config_dir   = dataset_config.get('config_dir','manuscript/manuscript_figures_data_config')

    cache = {}
    for b in batch_ids:
        config_name = config_fmt.format(batch=b)
        config_path_data = f'{config_dir}/{config_name}'
        config_data = load_config_file(config_path_data, 'data')
        config_data.OUTPUTS_FOLDER = path_to_embeddings

        X, y, _ = load_embeddings(path_to_embeddings, config_data)

        if multiplexed:
            analyzer = AnalyzerMultiplexMarkers(config_data, path_to_embeddings)
            X, y, _ = analyzer.calculate(X, y)

        cache[b] = (np.asarray(X), np.asarray(y).reshape(-1))
    if not cache:
        raise ValueError("No batches loaded.")
    return cache

def concat_from_cache(cache, batch_ids):
    """Concatenate (X,y) from the cache for the given batch_ids."""
    X_list, y_list = [], []
    for b in batch_ids:
        if b not in cache:
            raise KeyError(f"Batch {b} missing from cache.")
        X, y = cache[b]
        X_list.append(X); y_list.append(y)
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

def ensure_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple, set)): return [v for v in x]
    return [x]

def get_top_features(X, y, n_features=10):
    """
    Selects the top n features based on ANOVA F-value.
    """

    # Select top 100 features using F-score
    f_scores, _ = f_classif(X, y)
    top_features = np.argsort(f_scores)[-n_features:]
    return top_features

def train_and_evaluate_classifier(
    X_train, X_test, y_train_mapped, y_test_mapped,
    balance=False,
    norm=False,
    choose_features=False,
    top_k=100,
    apply_pca=False,
    pca_components=50,
    classifier_class=cuMLLogisticRegression,
    classifier_kwargs=dict(),
    return_proba=False
):
    # Balance
    if balance:
        ros = RandomOverSampler(random_state=42)
        X_train, y_train_mapped = ros.fit_resample(X_train, y_train_mapped)

    # Normalize
    if norm:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Feature selection
    if choose_features:
        print(f"Selecting top {top_k} features...")
        top_features = get_top_features(X_train, y_train_mapped, top_k)
        X_train = X_train[:, top_features]
        X_test = X_test[:, top_features]

    # PCA
    if apply_pca:
        pca = PCA(n_components=pca_components, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # Detect if model is GPU-based (cuML)
    is_gpu = 'cuml' in classifier_class.__module__.lower()

    # Move to GPU if needed
    if is_gpu:
        X_train = cudf.DataFrame.from_records(X_train)
        X_test = cudf.DataFrame.from_records(X_test)
        y_train_mapped = cudf.Series(y_train_mapped)

    # Train
    clf = classifier_class(**classifier_kwargs)
    clf.fit(X_train, y_train_mapped)

    # Predict
    y_pred = clf.predict(X_test)
    if hasattr(y_pred, "to_numpy"):
        y_pred = y_pred.to_numpy()
    report = classification_report(y_test_mapped, y_pred, output_dict=True)
    print(classification_report(y_test_mapped, y_pred))
    if hasattr(clf, "predict_proba") & return_proba:
        y_proba = clf.predict_proba(X_test)
    else:
        y_proba = None

    # Confusion matrix
    labels_in_test = sorted(set(y_test_mapped))
    cm = confusion_matrix(y_test_mapped, y_pred, labels=labels_in_test)
    return report['accuracy'], cm, y_pred, y_proba

def _append_results_csv(
    results_csv: str,
    *,
    dataset_config: dict,
    batches,
    train_specific_batches,
    test_specific_batches,
    classifier_class,
    classifier_kwargs,
    balance: bool,
    norm: bool,
    choose_features: bool,
    top_k: int,
    apply_pca: bool,
    pca_components: int,
    label_map,
    macro_stats: dict,       # dict from stats_df Macro Average
):
    """Append one summary row to results_csv (create-if-missing, append-if-exists)."""
    # flatten dataset_config -> ds_* columns
    ds_cols = {f"ds_{k}": v for k, v in dataset_config.items()}
    row = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        **ds_cols,
        "batches": json.dumps(list(batches)),
        "train_on": json.dumps([] if train_specific_batches is None else list(train_specific_batches)),
        "test_on": json.dumps([] if test_specific_batches is None else list(test_specific_batches)),
        "model": getattr(classifier_class, "__name__", str(classifier_class)),
        "params": json.dumps(classifier_kwargs or {}, sort_keys=True),
        "balance": bool(balance),
        "norm": bool(norm),
        "choose_features": bool(choose_features),
        "top_k": int(top_k),
        "apply_pca": bool(apply_pca),
        "pca_components": int(pca_components),
        "label_map_provided": label_map is not None,
        **{f"{k}": round(float(v), 3) for k, v in macro_stats.items()},
    }
    header = not os.path.exists(results_csv)
    pd.DataFrame([row]).to_csv(results_csv, mode="a", header=header, index=False)

def run_baseline_model(
    dataset_config,                # dict with paths/loading settings for embeddings
    batches=[1, 2, 3, 7, 8, 9,],   # list of batch IDs to include in the experiment
    balance=False,                 # whether to balance class distributions during training
    norm=False,                    # whether to normalize features before training
    choose_features=False,         # whether to select top features 
    top_k=100,                     # number of features to keep if choose_features=True
    apply_pca=False,               # whether to reduce dimensionality with PCA
    pca_components=50,             # number of PCA components if apply_pca=True
    label_map=None,                # optional mapping to merge/remap labels, e.g. {"WT":0,"KO":1}
    classifier_class=cuMLLogisticRegression, # classifier class to use (any sklearn/cuML-compatible estimator)
    classifier_kwargs=dict(),      # extra arguments for the classifier constructor (e.g. {"max_depth":10})
    test_specific_batches=None,    # int or list: which batches to use as test folds; None = default LOOCV
    train_specific_batches=None,   # int or list: which batches to use for training; None = complement of test
    return_proba=False,            # if True, return DataFrame of predicted probabilities along with metrics
    calculate_auc=False,           # if True, compute ROC AUC for the predictions
    results_csv=None               # if provided, append results to this CSV file
):
    accuracies = []
    accumulated_cm = None
    all_y_true = []; all_y_pred = [];  all_y_proba = []; fold_classes=[]

    print("Loading all batches...")
    cache = load_all_batches(batches, dataset_config)
    print("Batches loaded.")

    test_specific_batches  = ensure_list(test_specific_batches)
    train_specific_batches = ensure_list(train_specific_batches)

    # determine test folds
    if test_specific_batches:
        iter_tests = test_specific_batches
    else:
        # default LOOCV unless train is fixed
        iter_tests = [b for b in batches if b not in train_specific_batches]
        if train_specific_batches:
            iter_tests = [iter_tests]

    for test_batches in iter_tests:
        if isinstance(test_batches, int):
            test_batches = [test_batches]
        # train set: explicit or complement
        train_batches = train_specific_batches or [b for b in batches if b not in test_batches]
        train_batches = [b for b in train_batches if b not in test_batches]  # exclude overlaps

        print(f"Training on Batches: {train_batches}, Testing on: {test_batches}.")
        if not train_batches: raise ValueError(f"Empty train set for test {test_batches}.")
        if not test_batches: raise ValueError("Empty test set.")

        X_train, y_train = concat_from_cache(cache, train_batches)
        X_test,  y_test  = concat_from_cache(cache, test_batches)

        # Optionally filter based on label_map
        if label_map is not None:
            allowed_labels = set(label_map.keys())
            train_mask = np.isin(y_train, list(allowed_labels))
            test_mask = np.isin(y_test, list(allowed_labels))
            X_train, y_train = X_train[train_mask], y_train[train_mask]
            X_test, y_test = X_test[test_mask], y_test[test_mask]
            y_train = np.array([label_map[l] for l in y_train])
            y_test = np.array([label_map[l] for l in y_test])

        # Encode labels numerically
        le = LabelEncoder()
        y_train_mapped = le.fit_transform(y_train)
        y_test_mapped = le.transform(y_test)

        print(f"\n=== Batch {test_batches} ===")
        print("Train:", np.shape(X_train), "Labels:", np.unique(y_train_mapped))
        print("Test:", np.shape(X_test), "Labels:", np.unique(y_test_mapped))
        count_labels(y_train)

        accuracy, cm, y_pred, y_proba = train_and_evaluate_classifier(
            X_train, X_test, y_train_mapped, y_test_mapped,
            balance=balance,
            norm=norm,
            choose_features=choose_features,
            top_k=top_k,
            apply_pca=apply_pca,
            pca_components=pca_components,
            classifier_class=classifier_class,
            classifier_kwargs=classifier_kwargs,
            return_proba=return_proba or calculate_auc
        )
        accuracies.append(accuracy)

        # Confusion matrix
        accumulated_cm = cm if accumulated_cm is None else accumulated_cm + cm
        all_y_true.extend(y_test_mapped)
        all_y_pred.extend(y_pred)
        fold_classes.append(list(le.classes_)) # collect per-fold classes during the loop
        all_y_proba.append(_to_numpy_proba(y_proba))

    # Final summary
    print("\n=== Overall Accuracy ===")
    print(np.mean(accuracies), accuracies)
    if label_map is not None:
        # Invert label_map for display
        inv_map_full = defaultdict(list)
        for k, v in label_map.items():
            inv_map_full[v].append(k.replace('_Untreated', ''))
        display_labels = [' / '.join(inv_map_full[i]) for i in range(len(inv_map_full))]
    else:
        display_labels = [label.replace('_Untreated', '') for label in le.classes_]
    disp = ConfusionMatrixDisplay(confusion_matrix=accumulated_cm, display_labels=display_labels)    
    disp.plot(xticks_rotation=90)
    plt.title("Combined Confusion Matrix Across Batches")
    plt.tight_layout()
    plt.show()
        
    # Generate list of binary confusion matrices
    binary_cms = multilabel_confusion_matrix(all_y_true, all_y_pred, labels=range(len(le.classes_)))

    # Compute and print stats
    stats_df = compute_multilabel_metrics(binary_cms, labels=le.classes_)
    # Ensure correct display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    print("\n=== Evaluation Metrics ===")
    print(stats_df.to_string(index=False))
    macro = stats_df.loc[stats_df['Label'].eq('Macro Average')].drop(columns='Label').iloc[0].to_dict()

    if results_csv:
        _append_results_csv(
            results_csv,
            dataset_config=dataset_config,
            batches=batches,
            train_specific_batches=train_specific_batches,
            test_specific_batches=test_specific_batches,
            classifier_class=classifier_class,
            classifier_kwargs=classifier_kwargs,
            balance=balance,
            norm=norm,
            choose_features=choose_features,
            top_k=top_k,
            apply_pca=apply_pca,
            pca_components=pca_components,
            label_map=label_map,
            macro_stats=macro,
        )

    if calculate_auc:
        classes_global = sorted(set().union(*fold_classes))
        proba_all = np.vstack([_align_proba(p, cls, classes_global) for p, cls in zip(all_y_proba, fold_classes)])
        aucs = compute_overall_auc(np.array(all_y_true), proba_all)
    if return_proba:
        all_y_proba = np.vstack(all_y_proba)
        df_proba = pd.DataFrame(all_y_proba, columns=le.classes_)
        df_proba['predicted'] = le.inverse_transform(all_y_pred)
        df_proba['true'] = le.inverse_transform(all_y_true)
        return df_proba, macro
    else:
        return macro

def _to_numpy_proba(p):
    try:
        if isinstance(p, cp.ndarray): return cp.asnumpy(p)
    except Exception:
        pass
    try:
        if isinstance(p, (cudf.DataFrame, cudf.Series)): return p.to_pandas().to_numpy()
    except Exception:
        pass
    return np.asarray(p)
   
def _align_proba(proba_fold, classes_fold, classes_global):
    out = np.zeros((proba_fold.shape[0], len(classes_global)))
    idx = {c:i for i,c in enumerate(classes_global)}
    for j,c in enumerate(classes_fold):
        out[:, idx[c]] = proba_fold[:, j]
    return out

def compute_overall_auc(y_true_all, proba_all, print_results=True):
    """
    Pooled ROC-AUC over all folds/splits.

    Parameters
    ----------
    y_true_all : array-like, shape (N,)
        Ground-truth class ids (0..C-1) for all test samples across folds.
    proba_all  : array-like, shape (N, C) or (N,)
        Predicted class probabilities stacked across folds.
        Binary can be (N, 2) or (N,) = prob of positive class.
        NOTE: Assumes a consistent class order across folds.

    Returns
    -------
    dict
        For binary: {"binary": auc}
        For multiclass (C>=3): {
            "ovr_macro", "ovr_weighted",
            "ovo_macro", "ovo_weighted"
        }
    """
    y_true_all = np.asarray(y_true_all)
    proba_all = np.asarray(proba_all)

    # Binary convenience: accept (N,) or (N,2)
    if proba_all.ndim == 1:
        aucs = {"binary": float(roc_auc_score(y_true_all, proba_all))}
    elif proba_all.shape[1] == 2:
        aucs = {"binary": float(roc_auc_score(y_true_all, proba_all[:, 1]))}
    else:
        aucs = {
            "ovr_macro":    float(roc_auc_score(y_true_all, proba_all, multi_class="ovr", average="macro")),
            "ovr_weighted": float(roc_auc_score(y_true_all, proba_all, multi_class="ovr", average="weighted")),
            "ovo_macro":    float(roc_auc_score(y_true_all, proba_all, multi_class="ovo", average="macro")),
            "ovo_weighted": float(roc_auc_score(y_true_all, proba_all, multi_class="ovo", average="weighted")),
        }

    if print_results:
        print("\n=== ROC AUC ===")
        for k, v in aucs.items():
            print(f"{k}: {v:.4f}")
    return aucs


def run_train_test_split_baseline(
    dataset_config,
    batches=[1, 2, 3, 7, 8, 9, 10],
    balance=False,
    norm=False,
    choose_features=False,
    top_k=100,
    apply_pca=False,
    pca_components=50,
    classifier_class=cuMLLogisticRegression,
    classifier_kwargs=dict(),
    return_proba=False  # If True, return predicted probabilities
):
    # Load and encode
    X, y = load_batches(batches, dataset_config= dataset_config)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("Train dataset")
    print(np.shape(y_train), np.shape(X_train), np.unique(y_train))
    count_labels(y_train)

    print("Test dataset")
    print(np.shape(y_test), np.shape(X_test), np.unique(y_test))
    count_labels(y_test)

    # Train and evaluate
    accuracy, cm, y_pred, y_proba = train_and_evaluate_classifier(
        X_train, X_test, y_train, y_test,
        balance=balance,
        norm=norm,
        choose_features=choose_features,
        top_k=top_k,
        apply_pca=apply_pca,
        pca_components=pca_components,
        classifier_class=classifier_class,
        classifier_kwargs=classifier_kwargs,
        return_proba=return_proba
    )

    # Only plot if not too many classes
    if len(le.classes_) <= 15:
        plot_confusion_matrix(y_test, y_pred, le) 
    else:
        print(f"Skipping confusion matrix plot ({len(le.classes_)} classes).")
    print(f"\nAccuracy: {accuracy:.4f}")
    # Generate list of binary confusion matrices
    binary_cms = multilabel_confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
    # Compute and print stats
    stats_df = compute_multilabel_metrics(binary_cms, labels=le.classes_)
    # Ensure correct display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    print("\n=== Evaluation Metrics ===")
    print(stats_df.to_string(index=False))
    if return_proba:
        return y_proba

def spearman_gpu(X, y, top_n=100):
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
    # Load data
    X_all, y_all = load_batches(batches, dataset_config= dataset_config)
    print('Data loaded.')
    y_all = np.array(y_all)

    # Optional: Balance dataset
    if balance:
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
    
    plot_cluster_confusion_matrix(df_with_merged, cluster_col='merged_cluster', title_prefix=f"{method.upper()} ")

    return df, cluster_summary, label_summary

def plot_cluster_confusion_matrix(df, cluster_col='cluster', title_prefix=''):
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

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clusters)
    disp.plot(xticks_rotation=90)
    plt.title(f"{title_prefix}Confusion Matrix: True Label vs {cluster_col}")
    plt.xlabel("Cluster")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

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


def compute_multilabel_metrics(conf_matrices, labels=None):
    """
    Computes per-class and macro-averaged accuracy, sensitivity, specificity, PPV, NPV, and F1 score
    from a multilabel confusion matrix.

    Parameters:
        conf_matrices: np.ndarray of shape (n_classes, 2, 2)
        labels: optional list of class names

    Returns:
        pd.DataFrame with per-class and overall statistics
    """
    rows = []

    for i, cm in enumerate(conf_matrices):
        tn, fp, fn, tp = cm.ravel()

        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        ppv = tp / (tp + fp) if (tp + fp) else np.nan
        npv = tn / (tn + fn) if (tn + fn) else np.nan

        # F1 from precision/recall
        if np.isfinite(ppv) and np.isfinite(sens) and (ppv + sens) > 0:
            f1 = 2 * ppv * sens / (ppv + sens)
        else:
            f1 = np.nan

        label = labels[i] if labels is not None else str(i)
        rows.append({'Label': label, 'Accuracy': acc, 'Sensitivity': sens,
                     'Specificity': spec, 'PPV': ppv, 'NPV': npv, 'F1': f1})

    df = pd.DataFrame(rows)

    # Add macro-average row
    macro = df[['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1']].mean(skipna=True)
    macro_row = pd.DataFrame([['Macro Average', *macro.values.tolist()]], columns=df.columns)

    df = pd.concat([df, macro_row], ignore_index=True)

    return df

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
