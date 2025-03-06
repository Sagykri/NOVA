from sklearn.metrics import adjusted_rand_score
import logging
import numpy as np

def cluster_without_outliers(X:np.ndarray[float], n_clusters:int, outliers_fraction:float=0.1, n_init:int=10, random_state:int=42)->np.ndarray[int]:
  """Cluster X using KMeans version that knows to handle with outliers

  Args:
      X (Iterable[float]): The values
      n_clusters (int): Expected number of clusters
      outliers_fraction (float, optional): The max percentage within the data allowed for outliers. Defaults to 0.1.
      n_init (int, optional): Number of times the k-means algorithm is run with different centroid seeds. Defaults to 10.
      random_state (int, optional): The seed. Defaults to 42.

  Returns:
      np.ndarray[int]: The cluster number per sample
  """
  from k_means_constrained import KMeansConstrained
  
  size_min = int(len(X)*outliers_fraction)
  logging.info(f"[K Means Constrained clustering] size_min = {size_min}")

  if size_min*n_clusters > len(X):
      size_min = np.floor((len(X)/n_clusters)*0.5)
      logging.info(f"[K Means Constrained clustering] size_min was changed to  = {size_min} to support the number of clusters")
  
  clf = KMeansConstrained(
          n_clusters=n_clusters,
          size_min=size_min,
          n_init=n_init,
          random_state=random_state
      )
  predicted_labels = clf.fit_predict(X)
  return predicted_labels

def calc_ari_with_kmeans(X:np.ndarray[float], true_labels:np.ndarray[str], outliers_fraction:float=0.1)->float:
  """
  Calculate ARI score on data using kmeans as predicted labels
  Args:
      X (np.ndarray[float]): The data
      true_labels (np.ndarray[str]): list of strings 

  Returns:
      float: the ari score
  """       
  # K-means for standard (centroid-based) clustering
  k = np.unique(true_labels).shape[0]
  logging.info(f"[calc_clustering_validation_metric] unique_labels count: {k}, {np.unique(true_labels)}")

  kmeans_labels = cluster_without_outliers(X, n_clusters=k, outliers_fraction=outliers_fraction, n_init=10, random_state=42)
  ari = adjusted_rand_score(true_labels, kmeans_labels)
  ari = round(ari, 3)

  logging.info(f"[calc_clustering_validation_metric] ARI: {ari}")

  return ari

