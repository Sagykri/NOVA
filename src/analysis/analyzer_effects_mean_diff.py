import sys
import os
sys.path.insert(1, os.getenv("NOVA_HOME"))

import numpy as np
from typing import Tuple

from src.analysis.analyzer_effects import AnalyzerEffects
from src.datasets.dataset_config import DatasetConfig

class AnalyzerEffectsMeanDiff(AnalyzerEffects):
    def __init__(self, data_config: DatasetConfig, output_folder_path:str):
        super().__init__(data_config, output_folder_path)  

    def _compute_effect(self, group_baseline: np.ndarray[float], group_pert: np.ndarray[float]) -> Tuple[float,float]:
        """Compute the effect size (Cohen's d) and its estimated variance between two groups:
            baseline and perturbed, based on their distances to the baseline centroid.

            The method measures how far the perturbed samples deviate from the baseline centroid
            compared to the baseline distribution itself. It returns both the standardized mean
            difference (Cohen's d) and an estimate of its sampling variance.
        Args:
        group_baseline (np.ndarray[float]): Embeddings for the baseline group. 
                                            Used to compute the baseline centroid and the 
                                            distances of baseline samples to that centroid.
        group_pert (np.ndarray[float]):     Embeddings for the perturbed group.
                                            Used to compute the distances to the 
                                            baseline centroid.

        Returns:
            Tuple[float, float]: 
            - Cohen's d: the standardized difference in mean distance to the centroid 
              between the perturbed and baseline groups.
            - Variance of Cohen's d: an estimate of the sampling variance of the effect size.
        """
        centroid_baseline = np.mean(group_baseline, axis=0)
        dists_baseline = np.linalg.norm(group_baseline - centroid_baseline, axis=1)
        dists_pert = np.linalg.norm(group_pert - centroid_baseline, axis=1)
        
        std_baseline = dists_baseline.std(ddof=1)
        std_pert = dists_pert.std(ddof=1)
        
        # Pooled standard deviation
        n_baseline, n_pert = len(group_baseline), len(group_pert)
        pooled_std = np.sqrt(((n_baseline - 1) * std_baseline ** 2 + (n_pert - 1) * std_pert ** 2) / (n_baseline + n_pert - 2))

        mean_dist_pert = np.mean(dists_pert)
        mean_dist_baseline = np.mean(dists_baseline)
        
        # Cohen's d: difference in means / pooled std
        effect_size = (mean_dist_pert - mean_dist_baseline) / pooled_std
        effect_size_variance = ((n_baseline + n_pert)/(n_baseline * n_pert)) + ((effect_size**2) / (2*(n_baseline + n_pert)))
        return effect_size, effect_size_variance
    
    
    def _get_save_path(self, output_folder_path:str)->str:
        savepath_combined = os.path.join(output_folder_path, f"combined_effects.csv")
        savepath_batch = os.path.join(output_folder_path, f"batch_effects.csv")
        return savepath_combined, savepath_batch 