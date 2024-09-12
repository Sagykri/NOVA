from sklearn.metrics import adjusted_rand_score
import torch
import logging
import numpy as np

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

def calc_ari_with_kmeans(X, true_labels, outliers_fraction=0.1)->float:
  """
  Calculate ARI score on data using kmeans as predicted labels
  Args:
      X (n_samples, n_features): The data
      true_labels (array): list of strings 

  Returns:
      float: the ari score
  """       
  # K-means for standard (centroid-based) clustering
  k = np.unique(true_labels).shape[0]
  kmeans_labels = cluster_without_outliers(X, n_clusters=k, outliers_fraction=outliers_fraction, n_init=10, random_state=42)
  ari = adjusted_rand_score(true_labels, kmeans_labels)
  ari = round(ari, 3)

  logging.info(f"[calc_clustering_validation_metric] ARI: {ari}")

  return ari

