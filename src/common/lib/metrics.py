import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import tensorflow as tf

def calc_metrics(X, labels_true, n_clusters=3, seed=1):
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
  adjusted_rand_score_val, silhouette_score_val = calc_metrics(X, labels_true.reshape(-1,), n_clusters=n_clusters)

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
  
def calc_reconstruction_error(model, images_indexes=None, entire_data=None, embvecs=None,\
                      reset_embvec=True, only_second_layer=False, show=False,
                      cmap_original = 'rainbow', cmap_reconstructed = 'rainbow', enhance_contrast=True):
  """Calculate reconstruction error (MSE between input and reconstructed image)

  Args:
      model (_type_): The model
      images_indexes (_type_, optional): Which images to take. Defaults to None (take all).
      entire_data (_type_, optional): The entire data in the dataset (for normalizing the score). Defaults to None.
      embvecs (_type_, optional): Precomputed embedded vectors. Defaults to None (recalculate them).
      only_second_layer (bool, optional): Use only the second VQ. Defaults to False.
      show (bool, optional): Should we show the reconstructed image. Defaults to False.
      cmap_original (str, optional): cmap for plotting the input image. Defaults to 'rainbow'.
      cmap_reconstructed (str, optional): cmap for plotting the reconstructed image. Defaults to 'rainbow'.
      enhance_contrast (bool, optional): Should we enhance the contrast of the image. Defaults to True.

  Returns:
      _type_: The scores
  """
  
  images_reconstructed = model.generate_reconstructed_images(images_indexes=images_indexes, embvecs=embvecs,\
                      reset_embvec=reset_embvec, only_second_layer=only_second_layer, show=show,
                      cmap_original = cmap_original, cmap_reconstructed = cmap_reconstructed, enhance_contrast=enhance_contrast)
  images_input = model.analytics.data_manager.test_data[images_indexes]
  
  mse_imgs_norms = []
  
  for i in range(images_reconstructed.shape[-1]):
    squared = tf.square(images_reconstructed[...,i]-images_input[...,i])
    
    with tf.Session() as sess:
      squared_numpy = squared.eval()
      squared_imgs = [np.mean(squared_img) for squared_img in squared_numpy]
      mse_imgs_mean = np.mean(squared_imgs)
      mse_imgs_norm_self = mse_imgs_mean / np.var(images_input[...,i])
      mse_imgs_norm_all = mse_imgs_mean / np.var(entire_data[...,i]) if entire_data is not None else None
      mse_imgs_norms += [{'by_self': mse_imgs_norm_self, 'by_all': mse_imgs_norm_all, 'unnormalized': mse_imgs_mean}]
    
  return mse_imgs_norms
