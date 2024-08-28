import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig
from src.common.lib.metrics import cluster_without_outliers
from src.Analysis.analyzer_distances import AnalyzerDistances

class AnalyzerDistancesARI(AnalyzerDistances):
    def __init__(self, trainer_conf: TrainerConfig, data_conf: DatasetConfig):
        super().__init__(trainer_conf, data_conf)  

    def _compute_scores(self, cur_embeddings: np.ndarray, cur_labels: np.ndarray) -> pd.DataFrame:
        """Compute ARI scores
        Args:
            cur_embeddings (np.ndarray): embeddings to calculate scores on
            cur_labels (np.ndarray): labels of the embeddings to calculate scores on; should contain only 2 unique labels

        Returns:
            pd.DataFrame: dataframe containing the score
        """
        n_clusters = 2
        kmeans_labels = KMeans(n_clusters=n_clusters, random_state=1).fit_predict(cur_embeddings)
        kmeans_constrained_labels = cluster_without_outliers(cur_embeddings, n_clusters=n_clusters, outliers_fraction=0.1, n_init=10, random_state=1)

        score = pd.DataFrame({
            'ARI_KMeans': [adjusted_rand_score(cur_labels, kmeans_labels)],
            'ARI_KMeansConstrained': [adjusted_rand_score(cur_labels, kmeans_constrained_labels)],
        })

        return score

