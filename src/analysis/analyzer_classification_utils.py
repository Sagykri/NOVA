import datetime as dt
import json
import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from collections import defaultdict

from cuml.linear_model import LogisticRegression as cuMLLogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
import cudf
import cupy as cp
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    multilabel_confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, auc)
from sklearn.model_selection import train_test_split
import matplotlib.cm as cm
import colorcet as cc

from src.common.utils import load_config_file
from src.embeddings.embeddings_utils import load_embeddings
from src.analysis.analyzer_multiplex_markers import AnalyzerMultiplexMarkers

## pandas settings for better df display
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

def load_batches(batch_ids, dataset_config):
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

def ensure_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple, set)): return [v for v in x]
    return [x]

def plan_folds(
    batches,
    test_specific_batches=None,
    train_specific_batches=None,
    *,
    train_each_as_singleton=False,   # train on [b], test on others
):
    def _tolist(x):
        if x is None: return []
        return x if isinstance(x, (list, tuple)) else [x]

    batches = list(batches)
    test_specific_batches  = _tolist(test_specific_batches)
    train_specific_batches = _tolist(train_specific_batches)

    folds = []

    if train_each_as_singleton:
        base_train = train_specific_batches or batches
        for tb in base_train:
            others = [b for b in batches if b != tb]
            if test_specific_batches:
                others = [b for b in others if b in test_specific_batches]
            if not others:
                raise ValueError(f"No test batches when training on batch {tb}.")
            folds.append({"train":[tb], "test":others})
        return folds

    if test_specific_batches:
        iter_tests = test_specific_batches
    else:
        iter_tests = [b for b in batches if b not in train_specific_batches]
        if train_specific_batches:
            iter_tests = [iter_tests]  # fixed-train; test=complement

    for test_batches in iter_tests:
        test_list = test_batches if isinstance(test_batches, (list, tuple)) else [test_batches]
        train_list = train_specific_batches or [b for b in batches if b not in test_list]
        train_list = [b for b in train_list if b not in test_list]
        if not train_list: raise ValueError(f"Empty train set for test {test_list}.")
        if not test_list:  raise ValueError("Empty test set.")
        folds.append({"train":train_list, "test":test_list})

    return folds

def concat_from_cache(cache, batch_ids):
    """
    Concatenate (X,y) from the cache for the given batch_ids.
    Cach is a dict {batch_id: (X, y)} as returned by load_batches.
    """
    X_list, y_list = [], []
    for b in batch_ids:
        if b not in cache:
            raise KeyError(f"Batch {b} missing from cache.")
        X, y = cache[b]
        X_list.append(X)
        y_list.append(y)
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

def _filter_and_remap_labels(X_train, y_train, X_test, y_test, label_map):
    """
    Filter samples to allowed labels and remap via label_map.
    Matching is substring-based: if a label contains a key from label_map,
    it will be mapped to that alias.
    """
    if label_map is None:
        return X_train, y_train, X_test, y_test

    # function to remap a single label
    def remap(label):
        for k, v in label_map.items():
            if k in label:   # substring match
                return v
        return None  # if no match, drop it

    # remap all labels
    y_train_mapped = np.array([remap(l) for l in y_train])
    y_test_mapped  = np.array([remap(l) for l in y_test])

    # keep only those with a mapping
    train_mask = y_train_mapped != None
    test_mask  = y_test_mapped  != None

    X_train, y_train = X_train[train_mask], y_train_mapped[train_mask]
    X_test,  y_test  = X_test[test_mask],  y_test_mapped[test_mask]

    return X_train, y_train, X_test, y_test

def count_labels(y):
    counts = Counter(y)
    for label, count in counts.items():
        print(f"{label}: {count}")

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
    get_proba=True
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

    if get_proba:
        if hasattr(clf, "predict_proba"):
            y_scores = clf.predict_proba(X_test)
        elif hasattr(clf, 'decision_function'):
            y_scores = clf.decision_function(X_test)
        else:
            y_scores = None

    # Confusion matrix
    cm = confusion_matrix(y_test_mapped, y_pred, labels=sorted(set(y_test_mapped)))
    return cm, y_pred, y_scores

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

def compute_multilabel_metrics(conf_matrices, labels=None, overall_cm=None):
    """
    Computes per-class and macro-averaged accuracy, sensitivity, specificity, PPV, NPV, and F1 score
    from a multilabel confusion matrix.

    Parameters:
        conf_matrices: np.ndarray of shape (n_classes, 2, 2)
        labels: optional list of class names

    Returns:
        pd.DataFrame with per-class and overall statistics
         If `overall_cm` (K×K) is provided, an extra 'Correct / Total Accuracy' row is appended,
        where Accuracy = trace(overall_cm) / overall_cm.sum(), other cells are NaN.
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

    # Accuracy = correct predictions / total samples from the aggregated confusion matrix
    if overall_cm is not None and overall_cm.size > 0 and overall_cm.sum() > 0:
        overall_acc = float(np.trace(overall_cm) / overall_cm.sum())
    else:
        overall_acc = np.nan

    macro_row['Correct/Total Accuracy'] = overall_acc
    df['Correct/Total Accuracy'] = np.nan

    # Add macro row last
    df = pd.concat([df, macro_row], ignore_index=True)

    return df

def remove_untreated_from_labels(labels):
    """Remove '_Untreated' from a single label or a list/array of labels."""
    if isinstance(labels, str):
        return labels.replace('_Untreated', '')
    return [str(l).replace('_Untreated', '') for l in labels]

def plot_confusion_matrix(cm, labels, title="Confusion Matrix", cmap="Blues",
                          xlabel=None, ylabel=None, save_path=None,
                          show_percentages=True, annotate_threshold=50):
    """
    Plot a confusion matrix with automatic scaling of figure & font size.
    - Default: show raw counts.
    - If show_percentages=True: show row-wise % with raw counts in brackets.
      Only annotates diagonal cells and off-diagonals with count > annotate_threshold.
    """
    n_classes = len(labels)
    fig_size = np.round(1.8 * n_classes,1)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size)) 
    # scale fonts 
    font_size = int(round(23 * 26 / n_classes))
    font_size = max(8, min(font_size, 30))  # cap between 8 and 30

    print(f"Plotting confusion matrix with {n_classes} classes, fig_size={fig_size}, font_size={font_size}")

    # inside the annotation loop
    if show_percentages:
        # normalize row-wise
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_perc = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)

        # plot heatmap by percentage
        im = ax.imshow(cm_perc, cmap=cmap, vmin=0, vmax=1)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                if count > annotate_threshold or i == j:
                    perc = 100.0 * cm_perc[i, j]
                    text = f"{perc:.1f}%\n[{count}]"
                else:
                    text = "0"
                color = "white" if cm_perc[i, j] > 0.5 else "black"
                ax.text(j, i, text,
                        ha="center", va="center",
                        fontsize=font_size,
                        color=color)    
    else:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
                ax.text(j, i, str(count),
                        ha="center", va="center",
                        fontsize=font_size,
                        color=color)
                    
    font_size += 6

    # axis ticks
    ax.set_xticks(np.arange(n_classes))
    ax.set_xticklabels(labels, rotation=90, fontsize=font_size)
    ax.set_yticks(np.arange(n_classes))
    ax.set_yticklabels(labels, fontsize=font_size)

    # labels and title
    ax.set_title(title, fontsize=font_size+20, fontweight="bold", pad=30)
    ax.set_xlabel(xlabel if xlabel else "Predicted", fontsize=font_size+10, fontweight="bold")
    ax.set_ylabel(ylabel if ylabel else "True", fontsize=font_size+10, fontweight="bold")

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=font_size)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.show()

def _align_and_sum_confusions(cms, cm_classes):
    """
    Align per-fold confusion matrices (different class orders / missing classes)
    and sum them into a single multiclass confusion matrix.
    Returns: (global_cm, classes_global)
    """
    if not cms:
        return None, []

    # union of all class names across folds (keep deterministic order)
    classes_global = sorted(set().union(*[set(cls) for cls in cm_classes]))
    idx = {c: i for i, c in enumerate(classes_global)}
    k = len(classes_global)
    global_cm = np.zeros((k, k), dtype=int)

    for cm, cls in zip(cms, cm_classes):
        for i_true, c_true in enumerate(cls):
            gi = idx[c_true]
            for j_pred, c_pred in enumerate(cls):
                gj = idx[c_pred]
                global_cm[gi, gj] += cm[i_true, j_pred]
    return global_cm, classes_global

def _build_display_labels(label_map, classes_global, shorten=False):
    def _clean(x):
        return remove_untreated_from_labels(x) if shorten else str(x)

    if label_map is not None:
        inv_map_full = defaultdict(list)
        for k, v in label_map.items():
            inv_map_full[v].append(_clean(k))
        # Use the mapped class names (keys) as display labels
        display_labels = list(inv_map_full.keys())
    else:
        display_labels = [_clean(c) for c in classes_global]

    return display_labels

def _binary_cms_from_cm(cm):
    """
    Convert a KxK multiclass confusion matrix into K binary (2x2) matrices:
    for each class i: TP, FP, FN, TN.
    Shape: (K, 2, 2)
    """
    k = cm.shape[0]
    total = cm.sum()
    binary = np.zeros((k, 2, 2), dtype=int)
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    diag = np.diag(cm)

    for i in range(k):
        tp = diag[i]
        fn = row_sums[i] - tp
        fp = col_sums[i] - tp
        tn = total - tp - fp - fn
        binary[i, 1, 1] = tp
        binary[i, 1, 0] = fn
        binary[i, 0, 1] = fp
        binary[i, 0, 0] = tn
    return binary

def _average_stats_tables(stats_tables):
    """
    Average a list of stats DataFrames returned by compute_multilabel_metrics.
    Averages numeric columns per 'Label'. Non-numeric columns are ignored.
    """
    if not stats_tables:
        return pd.DataFrame()
    stats_all = pd.concat(stats_tables, ignore_index=True)
    num_cols = [c for c in stats_all.columns if c != "Label"]
    averaged = stats_all.groupby("Label", as_index=False)[num_cols].mean()

    # Move Macro Average to the end
    if "Macro Average" in averaged["Label"].values:
        macro = averaged[averaged["Label"] == "Macro Average"]
        averaged = averaged[averaged["Label"] != "Macro Average"]
        averaged = pd.concat([averaged, macro], ignore_index=True)

    return averaged

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
    macro_aggregated: dict,       # dict from stats_df aggregated Macro Average
    macro_avg: dict = None        # dict from stats_df_avg Macro Average (if available)
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
        **{f"{k}_aggregated": round(float(v), 3) for k, v in macro_aggregated.items()},
        **{f"{k}_avg": round(float(v), 3) for k, v in macro_avg.items()},
    }
    header = not os.path.exists(results_csv)
    pd.DataFrame([row]).to_csv(results_csv, mode="a", header=header, index=False)

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

def save_classification_df(df: pd.DataFrame, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

def plot_roc_ovr(y_true, y_proba, class_names=None, save_path=None, title="ROC Curve (OvR)"):
    """
    Plot One-vs-Rest ROC curves (and AUCs) for binary or multiclass classification.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    plt.figure()

    if y_proba.ndim == 1 or y_proba.shape[1] == 1:
        proba = y_proba if y_proba.ndim == 1 else y_proba[:, 0]
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    else:
        n_classes = y_proba.shape[1]
        y_bin = label_binarize(y_true, classes=range(n_classes))
        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]
        colors = cc.glasbey  # ~256 unique, printable, colorblind-friendly
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})", color=colors[i % len(colors)])

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=6.5,
        frameon=False
    )
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

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
    train_each_as_singleton=False,  # if True, train on each batch individually, test on all others
    get_proba=True,                # if True calculate the probabilities/scores of the predictions
    results_csv=None,               # if provided, append results to this CSV file
    plot_per_fold_cm=True,          # if True, plot confusion matrix for each fold
    save_path=None,                  # if provided, save plots to this path
):
    accumulated_cm = None
    all_y_true = []; all_y_pred = []; all_y_scores = []
    per_fold_stats = []; fold_classes=[]
    _cms = []   # collect per-fold CMs

    print("Loading all batches...")
    cache = load_batches(batches, dataset_config)
    print("Batches loaded.")

    test_specific_batches  = ensure_list(test_specific_batches)
    train_specific_batches = ensure_list(train_specific_batches)

    folds = plan_folds(
        batches=batches,
        test_specific_batches=test_specific_batches,
        train_specific_batches=train_specific_batches,
        train_each_as_singleton=train_each_as_singleton,  
        )
    
    for fold in folds:
        train_batches, test_batches = fold["train"], fold["test"]
        print(f"Training on Batches: {train_batches}, Testing on: {test_batches}.")

        X_train, y_train = concat_from_cache(cache, train_batches)
        X_test,  y_test  = concat_from_cache(cache, test_batches)

        # Optionally filter based on label_map
        if label_map is not None:
            X_train, y_train, X_test, y_test = _filter_and_remap_labels(
                    X_train, y_train, X_test, y_test, label_map)

        # Encode labels numerically
        le = LabelEncoder()
        y_train_mapped = le.fit_transform(y_train)
        y_test_mapped = le.transform(y_test)

        print(f"\n=== Fold (test={test_batches}) ===")
        print("Train:", np.shape(X_train), "Labels:", np.unique(y_train_mapped))
        print("Test:", np.shape(X_test), "Labels:", np.unique(y_test_mapped))
        count_labels(y_train)

        cm, y_pred, y_scores = train_and_evaluate_classifier(
            X_train, X_test, y_train_mapped, y_test_mapped,
            balance=balance, norm=norm,
            choose_features=choose_features, top_k=top_k,
            apply_pca=apply_pca, pca_components=pca_components,
            classifier_class=classifier_class, classifier_kwargs=classifier_kwargs,
            get_proba=get_proba
        )

        all_y_true.extend(y_test_mapped)
        all_y_pred.extend(y_pred)

        if y_scores is not None:
            all_y_scores.append(_to_numpy_proba(y_scores))
        
        # Collect per-fold CM (DON'T sum yet; align later)
        _cms.append(cm)
        fold_classes.append(list(le.classes_)) # collect per-fold classes during the loop
        
        # Per-fold metrics (from this fold only)
        bin_cms_fold = multilabel_confusion_matrix(
            y_test_mapped, y_pred, labels=range(len(le.classes_))
        )
        stats_df_fold = compute_multilabel_metrics(bin_cms_fold, labels=le.classes_, overall_cm=cm)
        per_fold_stats.append(stats_df_fold)
        if plot_per_fold_cm:
            print("\n=== Evaluation Metrics ===")
            print(stats_df_fold.to_string(index=False))
            plot_confusion_matrix(cm, le.classes_, 
                                  title="Confusion Matrix per Fold", 
                                  save_path = os.path.join(save_path, f"confusion_matrix_train-{'-'.join(map(str, train_batches))}_test-{'-'.join(map(str, test_batches))}.png")
                                                if save_path else None)
            if save_path:
                save_classification_df(stats_df_fold, save_path=os.path.join(save_path, f"metrics_train-{'-'.join(map(str, train_batches))}_test-{'-'.join(map(str, test_batches))}.csv"))

            if y_scores is not None:
                plot_roc_ovr(y_test_mapped, y_scores, class_names=le.classes_,
                             save_path = os.path.join(save_path, f"roc_curve_train-{'-'.join(map(str, train_batches))}_test-{'-'.join(map(str, test_batches))}.png")
                                                if save_path else None)


    # ---------- Align & Combine ----------
    accumulated_cm, classes_global = _align_and_sum_confusions(_cms, fold_classes)

    # ---------- Metrics ----------
    # Metrics from the aggregated confusion matrix (global)
    binary_cms = _binary_cms_from_cm(accumulated_cm)
    stats_df_agg = compute_multilabel_metrics(binary_cms, classes_global, accumulated_cm)
    # Averaged metrics across folds (mean of per-fold tables)
    stats_df_avg = _average_stats_tables(per_fold_stats)

    print("\n=== Evaluation Metrics (from aggregated confusion) ===")
    print(stats_df_agg.to_string(index=False))

    if not stats_df_avg.empty:
        print("\n=== Average Metrics Across Folds ===")
        print(stats_df_avg.to_string(index=False))

    if save_path:
        save_classification_df(stats_df_avg, 
                               save_path=os.path.join(save_path, f"metrics_avg.csv"))

    # Macros
    macro_aggregated = stats_df_agg.loc[stats_df_agg['Label'].eq('Macro Average')].drop(columns='Label').iloc[0].to_dict()
    macro_avg = (stats_df_avg.loc[stats_df_avg['Label'].eq('Macro Average')]
                            .drop(columns='Label').iloc[0].to_dict()) if not stats_df_avg.empty else {}
    
    # ---------- Plots ----------
    plot_confusion_matrix(accumulated_cm, classes_global, 
                          title="Combined Confusion Matrix Across Folds", 
                          save_path = os.path.join(save_path, f"confusion_matrix_train-all_folds.png")if save_path else None)
    # Overall ROC-AUC across all folds
    aligned_scores = np.vstack([_align_proba(p, cls, classes_global) for p, cls in zip(all_y_scores, fold_classes)])
    plot_roc_ovr(all_y_true, aligned_scores, class_names=classes_global,
                 save_path = os.path.join(save_path, f"roc_curve_all_folds.png") if save_path else None,
                 title="ROC Curve Across All Folds")   

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
            macro_aggregated=macro_aggregated,
            macro_avg= macro_avg
        )

    return macro_avg
    
def get_df_proba(y_proba, y_pred, y_true, le):
    y_proba = np.vstack(y_proba)
    df_proba = pd.DataFrame(y_proba, columns=le.classes_)
    df_proba['predicted'] = le.inverse_transform(y_pred)
    df_proba['true'] = le.inverse_transform(y_true)
    return df_proba

def create_confusion_matrix(y_true, y_pred, label_encoder, shorten_labels=False):
    """
    Plots a confusion matrix with correctly aligned labels and optional label cleaning.
    """
    # Get original class labels
    labels = list(label_encoder.classes_)

    # Clean labels if needed
    if shorten_labels:
        display_labels = remove_untreated_from_labels(labels)
    else:
        display_labels = labels

    # Compute confusion matrix using string labels
    y_true_str = label_encoder.inverse_transform(y_true)
    y_pred_str = label_encoder.inverse_transform(y_pred)
    cm = confusion_matrix(y_true_str, y_pred_str, labels=labels)

    return cm, display_labels

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
    return_proba=False,  # If True, return predicted probabilities
    label_map=None,
    save_path=None,
):
    # Load once via cache, then concat the requested batches
    cache = load_batches(batches, dataset_config)
    X, y = concat_from_cache(cache, batches)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Train dataset")
    print(np.shape(y_train), np.shape(X_train), np.unique(y_train))
    count_labels(y_train)

    print("Test dataset")
    print(np.shape(y_test), np.shape(X_test), np.unique(y_test))
    count_labels(y_test)

    # Optionally filter based on label_map
    if label_map is not None:
        X_train, y_train, X_test, y_test = _filter_and_remap_labels(
                X_train, y_train, X_test, y_test, label_map)
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Train and evaluate
    accuracy, cm, y_pred, y_proba, y_scores = train_and_evaluate_classifier(
        X_train, X_test, y_train_encoded, y_test_encoded,
        balance=balance,
        norm=norm,
        choose_features=choose_features,
        top_k=top_k,
        apply_pca=apply_pca,
        pca_components=pca_components,
        classifier_class=classifier_class,
        classifier_kwargs=classifier_kwargs,
        return_proba=return_proba,
    )

    # Only plot if not too many classes
    if len(le.classes_) <= 15:
        cm, dispalyed_labels = create_confusion_matrix(y_test_encoded, y_pred, le)
        plot_confusion_matrix(cm, dispalyed_labels, save_path=save_path) 
    else:
        print(f"Skipping confusion matrix plot ({len(le.classes_)} classes).")
    print(f"\nAccuracy: {accuracy:.4f}")
    # Generate list of binary confusion matrices
    binary_cms = multilabel_confusion_matrix(y_test_encoded, y_pred, labels=range(len(le.classes_)))
    # Compute and print stats
    stats_df = compute_multilabel_metrics(binary_cms, labels=le.classes_)

    print("\n=== Evaluation Metrics ===")
    print(stats_df.to_string(index=False))
    if save_path:
        save_classification_df(stats_df, save_path=os.path.join(save_path, "metrics.csv"))
    macro_avg = (stats_df.loc[stats_df['Label'].eq('Macro Average')]
                            .drop(columns='Label').iloc[0].to_dict()) if not stats_df.empty else {}
    return macro_avg
