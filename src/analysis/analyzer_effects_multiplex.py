import sys
import os
sys.path.insert(1, os.getenv("NOVA_HOME"))

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
import statsmodels.stats.meta_analysis as smm
from scipy.stats import norm, chi2
from typing import Tuple, Callable

from src.analysis.analyzer_effects_dist_ratio import AnalyzerEffectsDistRatio
from src.datasets.dataset_config import DatasetConfig

from src.common.utils import get_if_exists
from src.datasets.label_utils import get_batches_from_input_folders

class AnalyzerEffectsMultiplex(AnalyzerEffectsDistRatio):
    
    def __init__(self, data_config: DatasetConfig, output_folder_path:str):
        """Get an instance

        Args:
            data_config (DatasetConfig): The dataset configuration object. 
            output_folder_path (str): path to output folder
        """
        super().__init__(data_config, output_folder_path)
        self.feature_type = 'effects_multiplex'

    def calculate(self, embeddings:np.ndarray[float], labels:np.ndarray[str], n_boot:int=1000)->Tuple[pd.DataFrame, pd.DataFrame]:
        
        baseline_perturb_dict = self._get_baseline_perturb_dict()
        output_folder_path = self.get_saving_folder(feature_type=self.feature_type)
        logging.info(f'output folder path: {output_folder_path}')
        embeddings_df = self._prepare_embeddings_df(embeddings, labels, extract_marker=False, extract_rep=False)
        embeddings_dim = embeddings.shape[1]
        effects_df = self._calculate_all_effects(embeddings_df, baseline_perturb_dict, embeddings_dim, n_boot)
        combined_effects_df = self._combine_effects(effects_df, group_by_marker=False)
        self._correct_for_multiple_hypothesis(combined_effects_df)

        self.features = combined_effects_df, effects_df

        return combined_effects_df, effects_df

    def get_saving_folder(self, feature_type:str)->str:
        """Get the path to the folder where the features and figures can be saved
        Args:
            feature_type (str): string indicating the feature type ('distances','UMAP')
        """
        model_output_folder = self.output_folder_path
        feature_folder_path = os.path.join(model_output_folder, 'figures', self.data_config.EXPERIMENT_TYPE, feature_type)
        os.makedirs(feature_folder_path, exist_ok=True)
        
        input_folders = get_batches_from_input_folders(self.data_config.INPUT_FOLDERS)
        
        baseline_pertrub_dict = self._get_baseline_perturb_dict()
        # Flat the baseline-perturbation dictionary to get all unique cell lines and conditions
        cell_lines_conditions = np.unique([*baseline_pertrub_dict.keys(), *np.concatenate(list(baseline_pertrub_dict.values()))])
        title = f"{'_'.join(input_folders)}_{'_'.join(cell_lines_conditions)}"
        saveroot = os.path.join(feature_folder_path,f'{title}')
        
        return saveroot


    def _bootstrap_effect_size_variance(self, group_baseline:pd.DataFrame, group_pert:pd.DataFrame,
                                        effect_size_func:Callable[[np.ndarray[float],np.ndarray[float]],np.ndarray[float]],
                                        n_boot:int=1000, embeddings_dim:int=192, random_state:int=0, **kwargs)->float:
        """
        Estimates the variance of an effect size via bootstrapping.

        For each of `n_boot` bootstrap iterations, samples with replacement from both 
        baseline and perturbed groups, computes the effect size using `effect_size_func`, 
        and then returns the variance across bootstrap samples.

        Args:
            group_baseline (pd.DataFrame): DataFrame of shape (n_samples_baseline, n_features + metadata), 
                containing the baseline group embeddings and metadata.
            group_pert (pd.DataFrame): DataFrame of shape (n_samples_pert, n_features + metadata),
                containing the perturbed group embeddings and metadata.
            effect_size_func (Callable): Function that takes two arrays of shape 
                (n_boot, n_samples, n_features) and returns an array of effect sizes 
                (shape: n_boot,).
            n_boot (int): Number of bootstrap replicates (default: 1000).
            embeddings_dim (int): Dimensionality of the embedding vectors (default: 192).
            random_state (int): Random seed for reproducibility (default: 0).

        Returns:
            float: Estimated variance of the effect size from the bootstrap distribution.
        """

        logging.info("Original bootstrapping")

        # Extract embeddings arrays
        embeddings_baseline = group_baseline.iloc[:, :embeddings_dim].values
        embeddings_pert = group_pert.iloc[:, :embeddings_dim].values

        rng = np.random.default_rng(random_state)
        n_baseline = embeddings_baseline.shape[0]
        n_pert = embeddings_pert.shape[0]

        # Generate all bootstrap indices at once:
        boot_idx_baseline = rng.integers(0, n_baseline, size=(n_boot, n_baseline))
        boot_idx_pert = rng.integers(0, n_pert, size=(n_boot, n_pert))
        
        # Index bootstrap samples, result shape (n_boot, n_samples, features)
        group_baseline_boot = embeddings_baseline[boot_idx_baseline]
        group_pert_boot = embeddings_pert[boot_idx_pert]
        
        # Compute effect sizes for all bootstraps
        boot_effect_sizes = effect_size_func(group_baseline_boot, group_pert_boot)
        
        # Return variance estimate of effect size from bootstrap distribution
        return np.var(boot_effect_sizes, ddof=1) # Update 3.9.25 - Returning variance instead of std as expected by smm.combine_effects

    
    def _calculate_all_effects(self, embeddings_df: pd.DataFrame, baseline_perturb_dict, 
                               embeddings_dim:int, n_boot:int=1000) -> pd.DataFrame:
        results = []
        for baseline in baseline_perturb_dict:
            logging.info(f"[AnalyzerEffects] baseline: {baseline}")
            baseline_cell_line, baseline_cond = baseline.split('_')
            baseline_df = embeddings_df[
                                (embeddings_df.cell_line == baseline_cell_line) &
                                (embeddings_df.condition == baseline_cond)]
            for pert in baseline_perturb_dict[baseline]:
                logging.info(f"[AnalyzerEffects] pert: {pert}")
                pert_cell_line, pert_cond = pert.split('_')
                pert_df = embeddings_df[
                                (embeddings_df.cell_line == pert_cell_line) &
                                (embeddings_df.condition == pert_cond)]
                
                # Group each DataFrame by batch separately
                baseline_groups = baseline_df.groupby(['batch'])
                pert_groups = pert_df.groupby(['batch'])
                # Iterate over batch keys that appear in both baseline and perturbed
                common_batch_keys = set(baseline_groups.groups.keys()) & set(pert_groups.groups.keys())
                common_batch_keys = sorted(common_batch_keys)
                for batch in common_batch_keys:
                    logging.info(f"[AnalyzerEffects] batch: {batch}")
                    
                    batch_baseline_df = baseline_groups.get_group(batch)
                    batch_pert_df = pert_groups.get_group(batch)

                    min_required = self.data_config.MIN_REQUIRED
                    res = self._calculate_effect_per_unit(batch_baseline_df, 
                                                           batch_pert_df, embeddings_dim,
                                                           min_required, n_boot)
                    if res:
                        res.update({'batch': batch, 'baseline': baseline, 'pert': pert})
                        results.append(res)

        return pd.DataFrame(results)

    def _calculate_effect_per_unit(self, baseline_df:pd.DataFrame, 
                                    pert_df:pd.DataFrame, embeddings_dim: int, 
                                    min_required:int, n_boot:int=1000):
        """
        Calculate the effect size and variance between baseline and perturbed groups 
        within a single unit.

        Args:
            baseline_df (pd.DataFrame):    
                DataFrame of baseline group samples for a single unit,
                containing embedding columns plus metadata.
            pert_df (pd.DataFrame): 
                DataFrame of perturbed group samples for the same  unit.
            embeddings_dim (int): 
                Number of embedding feature columns.
            min_required (int): 
                Minimum sample size in each group required to calculate effect size (default: 1000).
            n_boot (int): 
                Number of bootstrap iterations for variance estimation (default: 1000).

        Returns:
            dict: Dictionary with keys:
                'baseline_size': Number of baseline samples,
                'perturb_size': Number of perturbed samples,
                'effect_size': Calculated effect size,
                'variance': Estimated variance of the effect size.
            Returns empty dict if sample sizes are below `min_required`.
        """
        n_baseline = baseline_df.shape[0]
        n_pert = pert_df.shape[0]
        if min(n_baseline, n_pert) < min_required:
            logging.warning(f"Too few samples: baseline={n_baseline}, pert={n_pert}. Minimum required is {min_required}")
            return {}
        logging.info(f"Sample size: baseline={n_baseline}, pert={n_pert}")

        effect_size, variance = self._compute_effect(baseline_df, pert_df, n_boot, embeddings_dim=embeddings_dim)

        return {'baseline_size':n_baseline, 'perturb_size':n_pert,
                 'effect_size': effect_size, 'variance':variance}
    
    
    def _get_baseline_perturb_dict(self) -> Dict[str, list[str]]:
        """
        Retrieve baseline-to-perturbation mapping from the data configuration.
        """
        baseline_dict = get_if_exists(self.data_config, 'BASELINE_PERTURB', None)
        assert baseline_dict is not None, "BASELINE_PERTURB dict in data config is None. Example: {'WT_Untreated': ['WT_stress']}"
        return baseline_dict