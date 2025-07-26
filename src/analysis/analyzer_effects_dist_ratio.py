import sys
import os
sys.path.insert(1, os.getenv("NOVA_HOME"))

import numpy as np
from typing import Tuple, Callable
import logging

from src.analysis.analyzer_effects import AnalyzerEffects
from src.datasets.dataset_config import DatasetConfig
class AnalyzerEffectsDistRatio(AnalyzerEffects):
    def __init__(self, data_config: DatasetConfig, output_folder_path:str):
        super().__init__(data_config, output_folder_path)  

    def _compute_effect(self, group_baseline: np.ndarray[float], 
                        group_pert: np.ndarray[float], n_boot:int=1000)->Tuple[float,float]:
        """Compute the effect size (Log2FC) and its estimated variance between baseline and 
        perturbed groups. The effect is computed as the log2 of the ratio between distances 
        to the baseline medoid.

        Args:
        group_baseline (np.ndarray[float]): Embeddings for the baseline group. 
        group_pert (np.ndarray[float]):     Embeddings for the perturbed group.
        n_boot (int):                       Number of bootstrap iterations for estimating 
                                            variance (default: 1000).

        Returns:
            Tuple[float, float]: 
            - Effect size: Log2FC between median distances to the 
                baseline medoid.
            - Variance of the effect size estimated via bootstrap.
        """
        effect_size = self._compute_effect_size_baseline_distance(group_baseline[np.newaxis, ...], group_pert[np.newaxis, ...])[0]
        logging.info(f'effect size: {effect_size}')
        # bootstrap variance
        effect_size_var = self._bootstrap_effect_size_variance(group_baseline, group_pert, self._compute_effect_size_baseline_distance,
                                                               n_boot=n_boot)
        
        return effect_size, effect_size_var
    
    
    def _get_save_path(self, output_folder_path:str)->str:
        savepath_combined = os.path.join(output_folder_path, f"combined_effects.csv")
        savepath_batch = os.path.join(output_folder_path, f"batch_effects.csv")
        return savepath_combined, savepath_batch 
    
    @staticmethod 
    def _compute_effect_size_baseline_distance(group_baseline:np.ndarray[float], 
                                               group_pert:np.ndarray[float])->np.ndarray[float]:
        """
        Computes Log2FC effect size between two groups based on distances to the baseline medoid.

        This function assumes each input is a batch of bootstrap replicates (n_boot, n_samples, features).
        For each replicate, it computes the medoid of the baseline group, calculates distances of 
        both baseline and perturbed samples to that medoid, and then computes the ratio between 
        their median distances. Eventually, apply Log2 on the ratio.

        Args:
            group_baseline (np.ndarray): Array of shape (n_boot, n_samples_baseline, n_features),
                representing bootstrap replicates of baseline group embeddings.
            group_pert (np.ndarray): Array of shape (n_boot, n_samples_pert, n_features),
                representing bootstrap replicates of perturbed group embeddings.

        Returns:
            np.ndarray: Array of shape (n_boot,), containing the effect size 
            for each bootstrap replicate.
        """
        centroid_baseline = np.median(group_baseline, axis=1, keepdims=True)
        # Distances to centroid for each bootstrap sample
        dists_baseline = np.linalg.norm(group_baseline - centroid_baseline, axis=2)  # (n_boot, n_baseline)
        dists_pert = np.linalg.norm(group_pert - centroid_baseline, axis=2)         # (n_boot, n_pert)

        median_dist_baseline = np.median(dists_baseline,axis=1)  # (n_boot,)
        median_dist_pert = np.median(dists_pert,axis=1)          # (n_boot,)
        
        effect_sizes = np.log2(median_dist_pert / median_dist_baseline)
        return effect_sizes
    
    @staticmethod
    def _bootstrap_effect_size_variance(group_baseline:np.ndarray[float], group_pert:np.ndarray[float],
                                        effect_size_func:Callable[[np.ndarray[float],np.ndarray[float]],np.ndarray[float]],
                                        n_boot:int=1000, random_state:int=0)->float:
        """
        Estimates the variance of an effect size via bootstrapping.

        For each of `n_boot` bootstrap iterations, samples with replacement from both 
        baseline and perturbed groups, computes the effect size using `effect_size_func`, 
        and then returns the variance across bootstrap samples.

        Args:
            group_baseline (np.ndarray): Array of shape (n_samples_baseline, n_features), 
                containing the baseline group embeddings.
            group_pert (np.ndarray): Array of shape (n_samples_pert, n_features), 
                containing the perturbed group embeddings.
            effect_size_func (Callable): Function that takes two arrays of shape 
                (n_boot, n_samples, n_features) and returns an array of effect sizes 
                (shape: n_boot,).
            n_boot (int): Number of bootstrap replicates (default: 1000).
            random_state (int): Random seed for reproducibility (default: 0).

        Returns:
            float: Estimated variance of the effect size from the bootstrap distribution.
        """
        rng = np.random.default_rng(random_state)
        n_baseline = group_baseline.shape[0]
        n_pert = group_pert.shape[0]
        
        # Generate all bootstrap indices at once:
        boot_idx_baseline = rng.integers(0, n_baseline, size=(n_boot, n_baseline))
        boot_idx_pert = rng.integers(0, n_pert, size=(n_boot, n_pert))
        
        # Index bootstrap samples, result shape (n_boot, n_samples, features)
        group_baseline_boot = group_baseline[boot_idx_baseline]
        group_pert_boot = group_pert[boot_idx_pert]
        
        # Compute effect sizes for all bootstraps
        boot_effect_sizes = effect_size_func(group_baseline_boot, group_pert_boot)
        
        # Return variance estimate of effect size from bootstrap distribution
        return np.std(boot_effect_sizes, ddof=1)
    