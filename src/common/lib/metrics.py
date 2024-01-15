import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import torch
import sklearn.cluster as cluster
import pandas as pd
import logging

def calculate_mse(inpt, reconstructed):
  """Calculate MSE

  Args:
      inpt (torch.tensor|nparray): Input images
      reconstructed (torch.tensor|nparray): Reconstructed images

  Returns:
      dict: {'target': MSE score, 'nuclues': MSE score} 
  """
  if not torch.is_tensor(inpt):
    inpt = torch.from_numpy(inpt)
  if not torch.is_tensor(reconstructed):
    reconstructed = torch.from_numpy(reconstructed)
    
  data_ch = ['target', 'nucleus']
  mses = {}
  for ii, ch in enumerate(data_ch):
      mses[ch] = torch.nn.functional.mse_loss(inpt[:, ii, ...], reconstructed[:, ii, ...])
  return mses

def cluster_without_outliers(X, n_clusters, outliers_fraction=0.1, n_init=10, random_state=42):
  from k_means_constrained import KMeansConstrained
  
  size_min = int(len(X)*outliers_fraction)
  logging.info(f"[K Means Constrained clustering] size_min = {size_min}")
  
  clf = KMeansConstrained(
          n_clusters=n_clusters,
          size_min=size_min,
          n_init=n_init,
          random_state=random_state
      )
  predicted_labels = clf.fit_predict(X)
  return predicted_labels

def calc_clustering_validation_metric(X, true_labels, metrics=['ARI'], outliers_fraction=0.1):
        """
        Give data and true labels to calculate a dictionary with clustering metrics

        Args:
            X (N,-1): The data
            true_labels (array): list of strings 
            metrics (list, optional): clustering metrics to calculate; defaults to ['ARI'].

        Returns:
            scores (dictionary): dictionary where the key is the name of the metric 
                                 and the value is the clustering score
        """
        scores = {}
        if 'ARI' in metrics:
            ###########################
            ### For ARI calculation ###
            ###########################
            
            # K-means for standard (centroid-based) clustering
            k = len(true_labels.unique())
            # logging.warn("!!! Using Kmeans clustering instead of KMeansConstrained until the installtion of the package is fixed")
            # kmeans_labels = cluster.KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
            kmeans_labels = cluster_without_outliers(X, n_clusters=k, outliers_fraction=outliers_fraction, n_init=10, random_state=42)
            ari = adjusted_rand_score(true_labels, kmeans_labels)
            scores['ARI'] = round(ari, 3)
            
            logging.info(f"[calc_clustering_validation_metric] ARI: {ari}")
            
        return scores

def get_metrics_figure(X, labels_true, savepath=None, ax=None, outliers_fraction=0.1):
  """Generate a plot displaying the metrics

  Args:
      X (array): The data
      labels_true ([string]): The true labels
      savepath (string, optional): Where to save the plot. Defaults to None.

  Returns:
      [Fig, ARI]: The figure and the score
  """
  scores = calc_clustering_validation_metric(X, pd.Series(labels_true.reshape(-1,)), metrics=['ARI'], outliers_fraction=outliers_fraction)

  n_scores = len(scores.keys())
  titles = ["ARI"]
  vranges = [(-0.5,1)]

  # Optional
  linecolor = 'r'
  linewidth = 5
  cmap = "Greys"

  if ax is None:
    _, ax = plt.subplots(1,n_scores, figsize=(5,0.2))

  for i, key in enumerate(scores.keys()):
    score = scores[key]
    vmin, vmax = vranges[i][0], vranges[i][1]
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=vmin, vmax=vmax)
    cax = ax[i] if n_scores > 1 else ax
    cb = plt.colorbar(sm, cax=cax, orientation='horizontal', pad=0.25)
    cb.set_ticks([score])
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_title(titles[i], fontsize=16) # Nancy for figure 2A - remove "ARI" text from scale bar
    cb.ax.plot([score]*2, [vmin,vmax], linecolor, linewidth=linewidth)
  
  if savepath is not None:
    plt.savefig(f"./umaps/{savepath}", dpi=300, facecolor="white",bbox_inches='tight')

  return ax, scores