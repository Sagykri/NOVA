import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import torch

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

# TODO: Adjust to torch version
def calc_clustering_validation(X, labels_true, n_clusters=3, seed=1):
  """Calc metrics (silhouette and ARI)

  Args:
      X (array): The data
      labels_true ([string]): The true labels
      n_clusters (int, optional): Number of expected classes. Defaults to 3.

  Returns:
      (ARI, silhouette score): The ARI and silhouette scores
  """
  labels_pred = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(X)  
  
  silhouette_score_val = silhouette_score(X, labels_pred)
  adjusted_rand_score_val = adjusted_rand_score(labels_true=labels_true, labels_pred=labels_pred)

  return adjusted_rand_score_val, silhouette_score_val

# TODO: Adjust to torch version
def plot_metrics(X, labels_true, n_clusters=3,savepath=None):
  """Generate a plot displaying the metrics

  Args:
      X (array): The data
      labels_true ([string]): The true labels
      n_clusters (int, optional): Number of expected classes. Defaults to 3.
      savepath (string, optional): Where to save the plot. Defaults to None.

  Returns:
      [ARI, silhouette]: The scores
  """
  adjusted_rand_score_val, silhouette_score_val = calc_clustering_validation(X, labels_true.reshape(-1,), n_clusters=n_clusters)

  scores = [adjusted_rand_score_val, silhouette_score_val]
  scores = [round(s,2) for s in scores]
  titles = ["ARI", "Silhouette Score"]
  vranges = [(-0.5,1), (-1,1)]

  # Optional
  linecolor = 'r'
  linewidth = 5
  cmap = "Greys"

  _, ax = plt.subplots(1,len(scores), figsize=(5,0.2))

  for i in range(len(scores)):
    score = scores[i]
    vmin, vmax = vranges[i][0], vranges[i][1]
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(sm, cax=ax[i], orientation='horizontal', pad=0.25)
    cb.set_ticks([score])
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_title(titles[i], fontsize=16)
    cb.ax.plot([score]*2, [vmin,vmax], linecolor, linewidth=linewidth)
  
  
  if savepath is not None:
    plt.savefig(f"./umaps/{savepath}", dpi=300, facecolor="white",bbox_inches='tight')
  
  plt.show()

  return scores