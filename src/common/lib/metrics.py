from sklearn.metrics import adjusted_rand_score
import torch
import logging
import numpy as np

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
            k = np.unique(true_labels).shape[0]
            kmeans_labels = cluster_without_outliers(X, n_clusters=k, outliers_fraction=outliers_fraction, n_init=10, random_state=42)
            ari = adjusted_rand_score(true_labels, kmeans_labels)
            scores['ARI'] = round(ari, 3)
            
            logging.info(f"[calc_clustering_validation_metric] ARI: {ari}")
            
        return scores

