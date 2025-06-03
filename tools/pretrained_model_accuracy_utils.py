import os
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

from src.datasets.label_utils import get_markers_from_labels
from src.figures.plotting_utils import save_plot

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from typing import List

def build_probs_dataframe(probs: np.ndarray, labels:List[str], unique_labels:List[str], anot_file_path: str) -> pd.DataFrame:
    """Build dataframe with softmax probabilities and true labels.

    Args:
        probs (np.ndarray): Softmax probabilities.
        labels (List[str]): full labels.
        unique_labels (List[str]): Unique full labels from dataset (to match between probabilty and label).
        anot_file_path (str): Path to annotation CSV.

    Returns:
        pd.DataFrame: DataFrame containing gene_name, true localization, and class-wise probabilities.
    """
    anot = pd.read_csv(anot_file_path)
    markers = get_markers_from_labels(labels)
    unique_markers = get_markers_from_labels(unique_labels)
    
    probs_df = pd.DataFrame(probs, columns=unique_markers)
    probs_df['gene_name'] = markers
    probs_df = probs_df.merge(anot, on='gene_name', how='left')
    probs_df = probs_df.rename(columns={'localization': "true_localization"})
    probs_df = probs_df.fillna('Other')
    return probs_df

def compute_localization_probs(probs_df: pd.DataFrame, anot: pd.DataFrame) -> pd.DataFrame:
    """Aggregate probabilities by localization categories.

    Args:
        probs_df (pd.DataFrame): DataFrame with per-protein probabilities and category annotations.
        anot (pd.DataFrame): Annotation table with gene_name(protein) and localization(category).

    Returns:
        pd.DataFrame: DataFrame of localization-summed probabilities + true labels (true_localization).
    """
    class_names = np.unique(anot.localization)
    sum_probs = pd.DataFrame(columns=class_names)
    markers_set = set(probs_df.columns) - {'gene_name', 'true_localization'}
    
    for localization, markers_group in anot.groupby('localization'):
        localization_markers = list(set(markers_group['gene_name']) & markers_set)
        sum_probs[localization] = probs_df[localization_markers].sum(axis=1)

    markers_with_no_localization = list(markers_set - set(anot.gene_name))
    sum_probs['Other'] = probs_df[markers_with_no_localization].sum(axis=1)

    sum_probs = sum_probs.merge(probs_df[['true_localization']], left_index=True, right_index=True)
    return sum_probs

def evaluate_classification_accuracy(sum_probs: pd.DataFrame, class_names: list, output_folder_path: str):
    """Evaluate and plot normalized confusion matrix.

    Args:
        sum_probs (pd.DataFrame): Summed probabilities by localization.
        class_names (list): List of true class labels.
        output_folder_path (str): Path to save the confusion matrix figure.
    """
    y_pred = sum_probs.drop(columns=['true_localization']).idxmax(axis=1)
    y_true = sum_probs['true_localization']

    logging.info(f'accuracy_score: {accuracy_score(y_true, y_pred):.4f}')

    cm = confusion_matrix(y_true, y_pred, normalize='true', labels=class_names)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    diagonal = np.diag(cm_df)

    diag_series = pd.Series(diagonal, index=cm_df.index)
    sorted_labels = diag_series.sort_values(ascending=False).index

    cm_sum_sorted = cm_df.loc[sorted_labels, sorted_labels]
    cm_sum_sorted = cm_sum_sorted.rename(columns={col:col.replace('_','\n').title() for col in cm_sum_sorted.columns},
                                     index={index:index.replace('_',' ').title() for index in cm_sum_sorted.index})

    fig = plt.figure(figsize=(8, 6))
    ax=sns.heatmap(cm_sum_sorted, annot=True, fmt='.2f', cmap='Blues', cbar=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Pre-trained model accuracy on Opencell data\n(normalized to true labels)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    logging.info(f'saving in {output_folder_path}/confusion_matrix')
    save_plot(fig, f'{output_folder_path}/confusion_matrix', dpi=100, save_png=True, save_eps=True)

def multiclass_roc(sum_probs: pd.DataFrame, class_names: list, output_folder_path: str):
    """
    Plot One-vs-Rest (OvR) ROC curves for multi-class classification.
    
    Args:
        sum_probs (pd.DataFrame): DataFrame with true labels and class probabilities.
        class_names (List[str]): List of all class names.
        output_folder_path (str): Directory to save the ROC plot.
    """
    y_score = sum_probs[class_names].values
    y_true = sum_probs["true_localization"].values
    y_true_bin = label_binarize(y_true, classes=class_names)

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig = plt.figure(figsize=(5,4), dpi=150)
    for i, class_name in enumerate(class_names):
        class_name = class_name.replace('_',' ').title()
        plt.plot(fpr[i], tpr[i], label=f'{class_name} (AUC = {roc_auc[i]:.4f})')

    handles, labels = plt.gca().get_legend_handles_labels()
    handles, labels = np.array(handles), np.array(labels)
    ordered_class_order = np.argsort(list(roc_auc.values()))[::-1]
    sorted_handles, sorted_labels = handles[ordered_class_order], labels[ordered_class_order]
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve (One-vs-Rest)')
    plt.legend(sorted_handles, sorted_labels, loc="lower right", fontsize='small')
    plt.tight_layout()
    save_plot(fig, f'{output_folder_path}/auc_curve', dpi=100, save_png=True, save_eps=True)

    macro_auc = roc_auc_score(y_true_bin, y_score, average='macro', multi_class='ovr')
    logging.info(f"Macro-average AUC: {macro_auc:.4f}")

    weighted_auc = roc_auc_score(y_true_bin, y_score, average='weighted', multi_class='ovr')
    logging.info(f"Weighted-average AUC: {weighted_auc:.4f}")

    micro_auc = roc_auc_score(y_true_bin, y_score, average='micro', multi_class='ovr')
    logging.info(f"Micro-average AUC: {micro_auc:.4f}")