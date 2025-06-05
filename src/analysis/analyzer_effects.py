
import sys
import os

sys.path.insert(1, os.getenv("NOVA_HOME"))

from abc import abstractmethod
import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import statsmodels.stats.meta_analysis as smm
from scipy.stats import norm

from src.datasets.dataset_config import DatasetConfig
from src.analysis.analyzer import Analyzer
from src.common.utils import get_if_exists

class AnalyzerEffects(Analyzer):
    """
    AnalyzerEffects is responsible for calculating distance metrics between different conditions
    based on model embeddings. The effects are computed for each marker and batch, comparing
    a baseline condition to other conditions.
    """
    def __init__(self, data_config: DatasetConfig, output_folder_path:str):
        """Get an instance

        Args:
            data_config (DatasetConfig): The dataset configuration object. 
            output_folder_path (str): path to output folder
        """
        super().__init__(data_config, output_folder_path)


    def calculate(self, embeddings:np.ndarray[float], labels:np.ndarray[str])->Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate effect sizes from embeddings and corresponding labels.

        Args:
            embeddings (np.ndarray): The embeddings array of shape (n_samples, n_features).
            labels (np.ndarray): Array of strings with metadata per sample in the format 'marker_cellline_condition_batch_rep'.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - combined_effects_df: DataFrame with overall (combined) effect sizes and pvalues per marker.
                - batch_effects_df: DataFrame with per-batch calculated effect sizes and pvalues.
        """
        baseline_perturb_dict = self._get_baseline_perturb_dict()
        embeddings_df = self._prepare_embeddings_df(embeddings, labels)
        embeddings_dim = embeddings.shape[1]
        batch_effects_df = self._calculate_all_effects(embeddings_df, baseline_perturb_dict, embeddings_dim)

        combined_effects_df = self._combine_effects(batch_effects_df, alt='two-sided')
        self.features = combined_effects_df, batch_effects_df

        return combined_effects_df, batch_effects_df
    
    def load(self)->None:
        """
        Load pre-calculated effects from a file into the self.features attribute.
        """
        output_folder_path = self.get_saving_folder(feature_type='effects')
        loadpath_combined, loadpath_batch = self._get_save_path(output_folder_path)
        self.features = pd.read_csv(loadpath_combined), pd.read_csv(loadpath_batch)
        return None

    def save(self)->None:
        """
        Save the calculated effects to a specified file.
        """
        output_folder_path = self.get_saving_folder(feature_type='effects')
        os.makedirs(output_folder_path, exist_ok=True)
        savepath_combined, savepath_batch = self._get_save_path(output_folder_path)
        logging.info(f"Saving combined effects to {savepath_combined}")
        self.features[0].to_csv(savepath_combined, index=False)
        logging.info(f"Saving batch effects to {savepath_batch}")
        self.features[1].to_csv(savepath_batch, index=False)
        return None

    @abstractmethod    
    def _compute_effect(self, group_baseline: np.ndarray[float], group_pert: np.ndarray[float]) -> Tuple[float, float]:
        """
        Abstract method to compute the effect between two groups of embeddings.

        Args:
            group_baseline (np.ndarray[float]): Embeddings of the baseline group.
            group_pert (np.ndarray[float]): Embeddings of the perturbed group.

        Returns:
             Tuple[float, float]: A tuple containing:
            - The effect (e.g., effect size)
            - The variance of the effect
        """
        pass

    def _combine_effects(self, batch_effects: pd.DataFrame, alt: str = "two-sided", 
                     effect_type: str = 'random', plot_forest: bool = False) -> pd.DataFrame:
        """
        Combine per-batch effect sizes into a single summary statistic per 
        marker using meta-analysis.

        Args:
            batch_effects : pd.DataFrame
                DataFrame containing per-batch effect sizes and variances, along with 'marker', 'baseline', and 'pert'.
            alt : str, default='two-sided'
                Type of statistical test to compute p-values. Must be one of {'two-sided', 'greater', 'smaller'}.
            effect_type : str, default='random'
                Type of meta-analysis to apply. Must be one of {'random', 'fixed'}.
                If 'random' fails due to variance estimation, the method falls back to 'fixed'.
            plot_forest : bool, default=False
                If True, generate forest plots for each marker (currently not implemented).

        Returns:
            pd.DataFrame
                A DataFrame with columns:
                ['marker', 'baseline', 'pert', 'combined_effect', 'pvalue', 'ci_low', 'ci_upp']
        """
        if alt not in {"two-sided", "greater", "smaller"}:
            raise ValueError("Parameter 'alt' must be one of: 'two-sided', 'greater', or 'smaller'")
        if effect_type not in {"random", "fixed"}:
            raise ValueError("Parameter 'effect_type' must be one of: 'random', 'fixed'")
        
        combined_effects = []
        for (marker, baseline, pert), marker_df in batch_effects.groupby(['marker','baseline','pert']):
            effects = marker_df['effect_size'].values
            variances = marker_df['variance'].values
            
            # # Run random effects meta-analysis
            meta_res = smm.combine_effects(effects, variances, method_re='dl') 
            
            summary = meta_res.summary_frame()
            effect_row = summary.loc[f"{effect_type} effect"]
            if (effect_row["sd_eff"] is np.nan or np.isnan(effect_row['sd_eff'])) and effect_type=='random':
                logging.warning(f"[AnalyzerEffects._combine_effects] for {marker} in {baseline} vs {pert}, Random effect model has invalid variance estimate — falling back to fixed effect.")
                effect_row = summary.loc[f"fixed effect"]
            combined_effect = effect_row["eff"]
            combined_se = effect_row["sd_eff"]
            ci_low = effect_row["ci_low"]
            ci_upp = effect_row["ci_upp"]
            
            # # Compute z and p
            z = combined_effect / combined_se
            if alt == "two-sided":
                pvalue = 2 * (1 - norm.cdf(abs(z)))
            elif alt == "greater":
                pvalue = 1 - norm.cdf(z)
            elif alt == "smaller":
                pvalue = norm.cdf(z)
            
            # if plot_forest: # TODO: if we want to save this add here..
            #     meta_res.plot_forest()
            #     plt.title(title)
            #     plt.show()

            combined_effects.append({'marker':marker, 'baseline':baseline,'pert':pert,
                'combined_effect':combined_effect, 'pvalue':pvalue,
                'ci_low':ci_low, 'ci_upp':ci_upp})
        return pd.DataFrame(combined_effects)

    def _permutation_test(self, observed, group_baseline, group_pert, 
                      n_permutations=1000, seed=42, alt="greater"):
        """
        Perform a permutation test for the observed effect size between two groups.

        Args:
            observed : float
                The observed test statistic (e.g., effect size).
            group_baseline : np.ndarray
                Embeddings for the baseline group.
            group_pert : np.ndarray
                Embeddings for the perturbed group.
            n_permutations : int, default=1000
                Number of permutations to perform.
            seed : int or None, default=42
                Seed for reproducibility.
            alt : str, default='greater'
                Type of test. One of {'two-sided', 'greater', 'less'}.

        Returns:
            float
                p-value from the permutation test.
        """
        if alt not in {"two-sided", "greater", "less"}:
            raise ValueError("Parameter 'alt' must be one of: 'two-sided', 'greater', or 'less'")

        if seed is not None:
            np.random.seed(seed)

        combined = np.vstack([group_baseline, group_pert])
        n_baseline = len(group_baseline)
        permuted = []
        for _ in range(n_permutations):
            perm = np.random.permutation(len(combined))
            perm_baseline = combined[perm[:n_baseline]]
            perm_pert = combined[perm[n_baseline:]]
            stat, _ = self._compute_effect(perm_baseline, perm_pert)
            permuted.append(stat)

        if alt == "greater":
            p_value = np.mean([x >= observed for x in permuted])
        elif alt == "smaller":
            p_value = np.mean([x <= observed for x in permuted])
        else: # two-sided test
            p_value = np.mean([abs(x) >= abs(observed) for x in permuted]) 

        return p_value
    
    def _calculate_effect_per_batch(self, batch_df: pd.DataFrame, baseline_cell_line: str, baseline_cond: str,
                                perturb_cell_line: str, perturb_cond: str, embeddings_dim: int):
        """
        Compute the effect size and variance between baseline and perturbed groups in a batch.

        Args:
            batch_df : pd.DataFrame
                DataFrame containing embeddings and associated metadata for one marker and one batch.
            baseline_cell_line : str
                Cell line identifier for the baseline.
            baseline_cond : str
                Condition identifier for the baseline.
            perturb_cell_line : str
                Cell line identifier for the perturbed.
            perturb_cond : str
                Condition identifier for the perturbation.
            embeddings_dim : int
                Dimensionality of the embedding vectors.

        Returns:
            dict
                Dictionary with keys: 
                ['baseline_size', 'perturb_size', 'effect_size', 'variance', 'pvalue'] 
                or an empty dict if insufficient samples.
        """
        baseline_df = batch_df[(batch_df.cell_line == baseline_cell_line) & (batch_df.condition == baseline_cond)]
        baseline_embeddings = baseline_df.iloc[:, :embeddings_dim].values

        perturb_df = batch_df[(batch_df.cell_line == perturb_cell_line) & (batch_df.condition == perturb_cond)]
        perturb_embeddings = perturb_df.iloc[:, 0:embeddings_dim].values
        if (baseline_embeddings.shape[0] < 2) or (perturb_embeddings.shape[0] <2):
            return {}
        effect_size, variance = self._compute_effect(baseline_embeddings, perturb_embeddings)
        pvalue = self._permutation_test(effect_size, baseline_embeddings, perturb_embeddings,)

        return {'baseline_size':len(baseline_df), 'perturb_size':len(perturb_df),
                 'effect_size': effect_size, 'variance':variance, 'pvalue':pvalue}

    def _get_baseline_perturb_dict(self) -> str:
        """
        Retrieve baseline-to-perturbation mapping from the data configuration.
        """
        baseline = get_if_exists(self.data_config, 'BASELINE_PERTURB', None)
        assert baseline is not None, "BASELINE_PERTURB is None. Example: {'WT_Untreated': ['WT_stress']}"
        return baseline

    def _prepare_embeddings_df(self, embeddings: np.ndarray[float], 
                           labels: np.ndarray[str]) -> pd.DataFrame:
        """
        Convert raw embeddings and sample labels into a structured DataFrame.

        Args:
            embeddings : np.ndarray[float]
                Embedding array with shape (n_samples, n_features).
            labels : np.ndarray[str]
                Array of label strings in the format: "marker_cellline_condition_batch_rep".

        Returns:
            pd.DataFrame
                DataFrame with embedding vectors and extracted metadata columns:
                ['marker', 'cell_line', 'condition', 'batch', 'rep'].
        """
        df = pd.DataFrame(embeddings)
        df['label'] = labels
        df[['marker', 'cell_line', 'condition', 'batch', 'rep']] = df.label.str.split('_', expand=True)
        return df
    
    def _calculate_all_effects(self, embeddings_df: pd.DataFrame, 
                           baseline_perturb_dict: Dict, embeddings_dim:int) -> pd.DataFrame:
        """
        Calculate batch-level effect sizes for all marker–baseline–perturbation combinations.

        Args:
            embeddings_df : pd.DataFrame
                DataFrame with embedding vectors and metadata, created by `_prepare_embeddings_df`.
            baseline_perturb_dict : dict
                Dictionary mapping each baseline (e.g., "WT_Untreated") to a list of perturbations.
            embeddings_dim : int
                Dimensionality of the embedding vectors.
        Returns:
            pd.DataFrame
                DataFrame with per-batch effect size statistics including:
                ['marker', 'baseline', 'pert', 'batch', 'baseline_size', 'perturb_size', 
                'effect_size', 'variance', 'pvalue'].
        """

        results = []
        for baseline in baseline_perturb_dict:
            logging.info(f"[AnalyzerEffects] baseline: {baseline}")
            baseline_cell_line, baseline_cond = baseline.split('_')

            for pert in baseline_perturb_dict[baseline]:
                logging.info(f"[AnalyzerEffects] pert: {pert}")
                pert_cell_line, pert_cond = pert.split('_')
                subset_df = embeddings_df[
                    embeddings_df.cell_line.isin([pert_cell_line, baseline_cell_line]) & embeddings_df.condition.isin([pert_cond, baseline_cond])
                ]
                for marker, marker_df in subset_df.groupby('marker'):
                    logging.info(f"[AnalyzerEffects] marker: {marker}")
                    for batch, batch_df in marker_df.groupby('batch'):
                        logging.info(f"[AnalyzerEffects] batch: {batch}")
                        res = self._calculate_effect_per_batch(batch_df, baseline_cell_line, baseline_cond, pert_cell_line, pert_cond, embeddings_dim)
                        if res:
                            res.update({'marker': marker, 'baseline': baseline, 'pert': pert, 'batch': batch})
                            results.append(res)
                        else:
                            logging.info(f'skipping {marker} in {batch} for missing samples')

        return pd.DataFrame(results)
        
