import sys
import os
sys.path.insert(1, os.getenv("NOVA_HOME"))

import numpy as np
import pandas as pd
from typing import Tuple, Callable
import logging
import scipy

from src.analysis.analyzer_effects import AnalyzerEffects
from src.datasets.dataset_config import DatasetConfig
class AnalyzerEffectsDistRatio(AnalyzerEffects):
    def __init__(self, data_config: DatasetConfig, output_folder_path:str):
        super().__init__(data_config, output_folder_path)  

    def _compute_effect(self, group_baseline: pd.DataFrame, group_pert: pd.DataFrame, n_boot:int=1000, embeddings_dim:int=192)->Tuple[float,float]:
        """Compute the effect size (Log2FC) and its estimated variance between baseline and 
        perturbed groups. The effect is computed as the log2 of the ratio between distances 
        to the baseline medoid.

        Args:
        group_baseline (pd.DataFrame): Embeddings and metadata for the baseline group. 
        group_pert (pd.DataFrame):     Embeddings and metadata for the perturbed group
        n_boot (int):                       Number of bootstrap iterations for estimating 
                                            variance (default: 1000).
        embeddings_dim (int):               Dimensionality of the embedding vectors (default: 192).

        Returns:
            Tuple[float, float]: 
            - Effect size: Log2FC between median distances to the 
                baseline medoid.
            - Variance of the effect size estimated via bootstrap.
        """

        baseline_embeddings = group_baseline.iloc[:, :embeddings_dim].values[np.newaxis, ...]
        perturb_embeddings = group_pert.iloc[:, :embeddings_dim].values[np.newaxis, ...]

        effect_size = self._compute_effect_size_baseline_distance(baseline_embeddings, perturb_embeddings)[0]
        logging.info(f'effect size: {effect_size}')
        # bootstrap variance
        effect_size_var = self._bootstrap_effect_size_variance(group_baseline, group_pert, self._compute_effect_size_baseline_distance,
                                                               n_boot=n_boot, embeddings_dim=embeddings_dim, 
                                                               subsample_fraction=self.data_config.SUBSAMPLE_FRACTION,
                                                               trimming_alpha=self.data_config.BOOTSTRAP_TRIMMING_ALPHA)
        
        return effect_size, effect_size_var
    
    def _compute_effect_size_baseline_distance(self, group_baseline:np.ndarray[float],
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
        
        effect_sizes = np.log2(median_dist_pert / (median_dist_baseline + np.finfo(float).eps))
        return effect_sizes

    def _bootstrap_effect_size_variance(self, group_baseline:pd.DataFrame, group_pert:pd.DataFrame,
                                        effect_size_func:Callable[[np.ndarray[float],np.ndarray[float]],np.ndarray[float]],
                                        n_boot:int=1000, embeddings_dim:int=192, random_state:int=0, **kwargs)->float:
        """
        Estimates the variance of an effect size via bootstrapping.

        For each of `n_boot` bootstrap iterations, samples with replacement from both 
        baseline and perturbed groups, computes the effect size using `effect_size_func`, 
        and then returns the variance across bootstrap samples.

        Args:
            group_baseline (pd.DataFrame): Embeddings and metadata for the baseline group.  
            group_pert (pd.DataFrame):     Embeddings and metadata for the perturbed group.
            effect_size_func (Callable): Function that takes two arrays of shape 
                (n_boot, n_samples, n_features) and returns an array of effect sizes 
                (shape: n_boot,).
            n_boot (int): Number of bootstrap replicates (default: 1000).
            embeddings_dim (int): Dimensionality of the embedding vectors (default: 192).
            random_state (int): Random seed for reproducibility (default: 0).
            subsample_fraction (float): Fraction of sites to sample in each bootstrap 
                iteration (default: 1, i.e. use all sites).
            trimming_alpha (float): Fraction of extreme values to trim from the 
                bootstrap effect size distribution (default: 0, i.e. no trimming. for example 0.01 means 1% from each tail).

        Returns:
            float: Estimated variance of the effect size from the bootstrap distribution.
        """
        subsample_fraction = kwargs.get('subsample_fraction', 1.0)
        trimming_alpha = kwargs.get('trimming_alpha', 0.0)

        rng = np.random.default_rng(random_state)

        # Extract embeddings arrays
        embeddings_baseline = group_baseline.iloc[:, :embeddings_dim].values
        embeddings_pert = group_pert.iloc[:, :embeddings_dim].values

        # Group by sites
        # Example: {'rep1_r03c09f55-ch2t1_panelJ_dNLS_processed.npy': array([ 0,  5,  9, 15, 21, 22, 27, 36, 46, 48, 50, 58, 75])}
        baseline_sites_groups = group_baseline.groupby("site", sort=False).indices 
        pert_sites_groups = group_pert.groupby("site", sort=False).indices

        def __generate_bootstrap_resamples(sites_groups: dict, n_boot: int, subsample_fraction:float) -> np.ndarray[int]:
            """Generates bootstrap indices by sampling one tile per site with replacement.
            Args:
                sites_groups (dict): Dictionary mapping site names to arrays of tile indices.
                    for example: {'rep1_r03c09f55-ch2t1_panelJ_dNLS_processed.npy': array([ 0,  5,  9, 15, 21, 22, 27, 36, 46, 48, 50, 58, 75])}
                n_boot (int): Number of bootstrap replicates.
                subsample_fraction (float): Fraction of sites to sample in each bootstrap iteration.
            Returns:
                np.ndarray[int]: Array of shape (n_boot, n_sites_sampled) containing
                                  bootstrap indices.
            """

            assert 0 < subsample_fraction <= 1, "SUBSAMPLE_FRACTION must be > 0 and <= 1"

            n_sites_to_sample = int(len(sites_groups.keys())**subsample_fraction)
            logging.info(f"[Analyzer_effects_dist_ratio._bootstrap_effect_size_variance] #sites to sample: {n_sites_to_sample} (out of {len(sites_groups.keys())} sites)")

            assert n_sites_to_sample > 1, "#sites to sample must be > 1"

            result = np.empty((n_boot, n_sites_to_sample), dtype=int)
            for b in range(n_boot):

                # Pick one tile from each site
                unique_sites_with_single_tile = {site_name: rng.choice(ti, size=1)[0] for site_name, ti in sites_groups.items()}

                # Select sites with replacement
                chosen_sites = rng.choice(list(unique_sites_with_single_tile.keys()), size=n_sites_to_sample, replace=True)
                chosen_tiles = [unique_sites_with_single_tile[site] for site in chosen_sites]

                result[b] = chosen_tiles

            return result

        def __log_extreme_tails(values):
            def __detect_extreme_tails_IQR(values, factor=3.0):
                """
                Flag if 1%/99% quantiles are 'far' compared to IQR.
                factor: how many times bigger than the IQR counts as extreme
                """
                values = np.asarray(values, float)
                values_sorted = np.sort(values)

                min_val, max_val = values_sorted[0], values_sorted[-1]
                q25, q75 = np.percentile(values_sorted, [25, 75])
                iqr = q75 - q25
                lower_gap = (q25 - min_val) / iqr if iqr > 0 else np.inf
                upper_gap = (max_val - q75) / iqr if iqr > 0 else np.inf
                is_extreme = (lower_gap > factor) or (upper_gap > factor)

                results = {'upper_outlier':{}, 'lower_outlier':{}}
                if lower_gap >= factor:
                    results['lower_outlier']['value'] = min_val
                    results['lower_outlier']['gap'] = q25 - min_val
                if upper_gap >= factor:
                    results['upper_outlier']['value'] = max_val
                    results['upper_outlier']['gap'] = max_val - q75

                return is_extreme, results

            def __detect_extreme_tails_MAD(values, z_score_threshold=3.5):
                values = np.asarray(values, float)
                # Explanation:
                # Calculating diffs = np.abs(values - np.median(values))
                # Checking for outliers diffs / np.median(diffs) > threshold
                outliers = (np.abs(values - np.median(values)) / scipy.stats.median_abs_deviation(values, scale="normal") > z_score_threshold)
                values = values[outliers]
                is_extreme = len(values) > 0

                return is_extreme, values.tolist()

            is_extreme_factor = 3.0
            is_extreme, extreme_details = __detect_extreme_tails_IQR(boot_effect_sizes, factor=is_extreme_factor)
            if is_extreme:
                logging.warning(f"[Analyzer_effects_dist_ratio._bootstrap_effect_size_variance] [IQR] Detected extreme tails (factor={is_extreme_factor}) in bootstrap effect sizes! Details: {extreme_details}")
            is_extreme_threshold = 3.5
            is_extreme, extreme_details = __detect_extreme_tails_MAD(boot_effect_sizes, z_score_threshold=is_extreme_threshold)
            if is_extreme:
                logging.warning(f"[Analyzer_effects_dist_ratio._bootstrap_effect_size_variance] [MAD] Detected extreme tails (factor={is_extreme_threshold}) in bootstrap effect sizes! Details: {extreme_details}")


        # Get one tile per site for each bootstrap replicate
        boot_idx_baseline = __generate_bootstrap_resamples(baseline_sites_groups, n_boot, subsample_fraction) # (n_boot, n_sites_sampled) 
        boot_idx_pert     = __generate_bootstrap_resamples(pert_sites_groups, n_boot, subsample_fraction) # (n_boot, n_sites_sampled)
        
        # Index bootstrap samples, result shape (n_boot, n_samples, features)
        group_baseline_boot = embeddings_baseline[boot_idx_baseline]
        group_pert_boot = embeddings_pert[boot_idx_pert]
        
        # Compute effect sizes for all bootstraps
        boot_effect_sizes = effect_size_func(group_baseline_boot, group_pert_boot)

        # Check for extreme tails and log
        __log_extreme_tails(boot_effect_sizes)

        # Trim outliers
        if trimming_alpha > 0:
            low, high = np.percentile(boot_effect_sizes, [100.0*trimming_alpha, 100.0*(1-trimming_alpha)])
            logging.info(f"[Analyzer_effects_dist_ratio._bootstrap_effect_size_variance] Trimming bootstrap effects (alpha: {trimming_alpha}, low: {low}, high: {high})")
            logging.info(f"[Analyzer_effects_dist_ratio._bootstrap_effect_size_variance] #effects before trimming: {boot_effect_sizes.shape}")
            boot_effect_sizes = boot_effect_sizes[(boot_effect_sizes >= low) & (boot_effect_sizes <= high)]
            logging.info(f"[Analyzer_effects_dist_ratio._bootstrap_effect_size_variance] #effects after trimming: {boot_effect_sizes.shape}")

            __log_extreme_tails(boot_effect_sizes)
        else:
            logging.info(f"[Analyzer_effects_dist_ratio._bootstrap_effect_size_variance] No trimming applied (alpha: {trimming_alpha})")

        # Return variance estimate of effect size from bootstrap distribution
        return np.var(boot_effect_sizes, ddof=1) # Update 3.9.25 - Returning variance instead of std as expected by smm.combine_effects
 