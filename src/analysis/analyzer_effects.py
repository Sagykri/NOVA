
import sys
import os

sys.path.insert(1, os.getenv("NOVA_HOME"))

from abc import abstractmethod
import logging
import pandas as pd
import numpy as np
from typing import Tuple
import statsmodels.stats.meta_analysis as smm
from scipy.stats import norm, chi2, t
from statsmodels.stats.multitest import multipletests
import math

from src.datasets.dataset_config import DatasetConfig
from src.datasets.label_utils import get_batches_from_input_folders
from src.analysis.analyzer import Analyzer
from src.common.utils import get_if_exists
from scipy import stats, optimize

class AnalyzerEffects(Analyzer):
    """
    AnalyzerEffects calculates effect sizes representing differences between baseline and perturbed groups
    using model embeddings. Effects are computed per marker and batch, then combined via meta-analysis.

    The main workflow:
    - Prepare a DataFrame of embeddings and metadata parsed from sample labels.
    - For each baseline-perturbation pair, calculate effect sizes within each marker-batch group.
    - Aggregate batch-level effects into overall marker-level effects using meta-analysis.
    """
    def __init__(self, data_config: DatasetConfig, output_folder_path:str):
        """Get an instance

        Args:
            data_config (DatasetConfig): The dataset configuration object. 
            output_folder_path (str): path to output folder
        """
        super().__init__(data_config, output_folder_path)
        self.feature_type = 'effects'

    def calculate(self, embeddings:np.ndarray[float], labels:np.ndarray[str], paths: np.ndarray[str], n_boot:int=1000)->Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate effect sizes from embeddings and corresponding labels.

        Args:
            embeddings (np.ndarray): 
                The embeddings array of shape (n_samples, n_features).
            labels (np.ndarray):
                Array of strings with metadata per sample in the format 
                'marker_cellline_condition_batch_rep'.
            paths (np.ndarray):
                Array of strings with file paths corresponding to each sample.
            n_boot (int):
                Number of bootstrap iterations used to estimate variance of
                effect sizes (default: 1000).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                combined_effects_df: Summary DataFrame aggregated across batches, with one row per marker-baseline-perturb pair.
                    Contains columns:
                        - 'marker': marker identifier
                        - 'baseline': baseline condition string (e.g. "WT_Untreated")
                        - 'pert': perturbation condition string (e.g. "WT_stress")
                        - 'combined_effect': meta-analyzed effect size estimate
                        - 'combined_se': standard error of the combined effect
                        - 'pvalue': p-value testing effect significance under the chosen alternative hypothesis
                        - 'ci_low', 'ci_upp': confidence interval bounds for the combined effect
                        - 'p_heterogeneity', 'I2', 'Q': statistics describing between-batch heterogeneity

                batch_effects_df: Detailed DataFrame with effect sizes and variances estimated per batch and marker.
                    Each row corresponds to a single batch-marker-baseline-perturb combination and includes:
                        - 'marker', 'baseline', 'pert', 'batch'
                        - 'baseline_size', 'perturb_size': number of samples in baseline and perturbed groups
                        - 'effect_size': bootstrap-estimated effect size for that batch
                        - 'variance': bootstrap-estimated variance of the effect size
        """
        output_folder_path = self.get_saving_folder(feature_type=self.feature_type)
        logging.info(f'output folder path: {output_folder_path}')
        embeddings_df = self._prepare_embeddings_df(embeddings, labels, paths)
        embeddings_dim = embeddings.shape[1]
        batch_effects_df = self._calculate_all_effects(embeddings_df, embeddings_dim, n_boot)
        combined_effects_df = self._combine_effects(batch_effects_df)
        self._correct_for_multiple_hypothesis(combined_effects_df)

        self.features = combined_effects_df, batch_effects_df

        return combined_effects_df, batch_effects_df
    
    def load(self)->None:
        """
        Load pre-calculated effects from a file into the self.features attribute.
        """
        output_folder_path = self.get_saving_folder(feature_type=self.feature_type)
        loadpath_combined, loadpath_batch = self._get_save_path(output_folder_path)
        self.features = pd.read_csv(loadpath_combined), pd.read_csv(loadpath_batch)
        return None

    def save(self)->None:
        """
        Save the calculated effects to a specified file.
        """
        output_folder_path = self.get_saving_folder(feature_type=self.feature_type)
        os.makedirs(output_folder_path, exist_ok=True)
        savepath_combined, savepath_batch = self._get_save_path(output_folder_path)
        
        logging.info(f"Saving combined effects to {savepath_combined}")
        self.features[0].to_csv(savepath_combined, index=False)
        
        logging.info(f"Saving batch effects to {savepath_batch}")
        self.features[1].to_csv(savepath_batch, index=False)
        
        return None

    def get_saving_folder(self, feature_type:str)->str:
        """Get the path to the folder where the features and figures can be saved
        Args:
            feature_type (str): string indicating the feature type ('distances','UMAP')
        """
        model_output_folder = self.output_folder_path
        feature_folder_path = os.path.join(model_output_folder, 'figures', self.data_config.EXPERIMENT_TYPE, feature_type)
        os.makedirs(feature_folder_path, exist_ok=True)
        
        input_folders = get_batches_from_input_folders(self.data_config.INPUT_FOLDERS)
        reps = self.data_config.REPS if self.data_config.REPS else ['all_reps']
        markers = get_if_exists(self.data_config, 'MARKERS', None)
        if markers is not None and len(markers)<=3:
            title = f"{'_'.join(input_folders)}_{'_'.join(reps)}_{'_'.join(markers)}"
        else:
            excluded_markers = list(self.data_config.MARKERS_TO_EXCLUDE) if self.data_config.MARKERS_TO_EXCLUDE else ["all_markers"]
            if excluded_markers != ['all_markers']:
                excluded_markers.insert(0,"without")
            title = f"{'_'.join(input_folders)}_{'_'.join(reps)}_{'_'.join(excluded_markers)}"
        baseline = self.data_config.BASELINE
        pert = self.data_config.PERTURBATION
        title= f'{baseline}_vs_{pert}_{title}'
        saveroot = os.path.join(feature_folder_path,f'{title}')
        return saveroot

    def _get_save_path(self, output_folder_path:str)->str:
        savepath_combined = os.path.join(output_folder_path, f"combined_effects.csv")
        savepath_batch = os.path.join(output_folder_path, f"batch_effects.csv")
        return savepath_combined, savepath_batch 

    
    @abstractmethod    
    def _compute_effect(self, group_baseline: np.ndarray[float], group_pert: np.ndarray[float],
                        n_boot:int=1000, embeddings_dim:int=192)->Tuple[float, float]:
        """
        Abstract method to compute the effect between two groups of embeddings, and its 
        estimated variance.

        Args:
            group_baseline (np.ndarray[float]): Embeddings of the baseline group.
            group_pert (np.ndarray[float]):     Embeddings of the perturbed group.
            n_boot (int):                       Number of bootstrap iterations for estimating 
                                                variance (default: 1000).
            embeddings_dim (int):               Dimensionality of the embedding vectors (default: 192).

        Returns:
             Tuple[float, float]: A tuple containing:
            - The effect (e.g., effect size)
            - The variance of the effect
        """
        pass

    def _correct_for_multiple_hypothesis(self, combined_effects_df: pd.DataFrame) -> None:
        """Correct p-values for multiple hypothesis testing. In-place adding columns to the dataframe:
            - 'adj_pvalue': adjusted p-values for 'pvalue' column
            - 'adj_p_heterogeneity': adjusted p-values for 'p_heterogeneity' column 
        Args:
            combined_effects_df (pd.DataFrame): The input dataframe with columns for adjusted pvalues: 
            'adj_pvalue', 'adj_p_heterogeneity'.
        """
        for pval_col in ['pvalue','p_heterogeneity']:
            if pval_col not in combined_effects_df.columns:
                continue

            combined_effects_df[pval_col] = combined_effects_df[pval_col].replace(0, np.finfo(float).eps)  # avoid log(0)
            _, adj_pvals, _, _ = multipletests(combined_effects_df[pval_col], method='fdr_bh')
            combined_effects_df[f'adj_{pval_col}'] = adj_pvals

    def _combine_effects(self, batch_effects: pd.DataFrame, alt: str = "greater", 
                     effect_type: str = 'random', group_by_marker:bool=True) -> pd.DataFrame:
        """
        Combine per-batch effect sizes into a single summary statistic using meta-analysis.
        Uses statsmodels' `combine_effects` with random effects by default.
        Falls back to fixed effects if random effects variance is invalid.

        Args:
            batch_effects (pd.DataFrame):       DataFrame containing per-batch effect sizes and 
                                                variances, with columns:
                                                ['marker' if group_by_marker, 'baseline', 'pert', 'effect_size', 'variance', 'batch'].
            alt (str, Optional):                Statistical alternative hypothesis for p-value 
                                                calculation. One of {'two-sided', 'greater', 'smaller'}.
            effect_type (str, Optional):        Meta-analysis model type: 'random' (default) or 'fixed'.
            group_by_marker (bool, Optional):   If True, combine effects per marker (in addition to baseline and pert). (Default: True)

        Returns:
            pd.DataFrame:                       DataFrame with combined effect statistics per marker, including columns:
                                                ['marker', 'baseline', 'pert', 'combined_effect', 
                                                'combined_se', 'pvalue', 'ci_low', 'ci_upp', 
                                                'p_heterogeneity', 'I2', 'Q'].
        """
        if alt not in {"two-sided", "greater", "smaller"}:
            raise ValueError("Parameter 'alt' must be one of: 'two-sided', 'greater', or 'smaller'")
        if effect_type not in {"random", "fixed"}:
            raise ValueError("Parameter 'effect_type' must be one of: 'random', 'fixed'")
        
        groupby = ['baseline', 'pert']
        if group_by_marker:
            groupby.append('marker')

        combined_effects = []
        for groups, df in batch_effects.groupby(groupby):
            baseline, pert = groups[0], groups[1] 

            if group_by_marker:
                marker = groups[2]

            effects = df['effect_size'].values
            variances = df['variance'].values

            # Run random effects meta-analysis
            meta_res = smm.combine_effects(effects.astype(np.float64), variances.astype(np.float64)) # Sagy 7.9.25 (statsmodels suitable for the default np dtype (float64). For stabilizing tau2 calculation in meta-analysis)

            summary = meta_res.summary_frame()
            tau2 = meta_res.tau2
            effect_row = summary.loc[f"{effect_type} effect"]
            if effect_type == 'random' and tau2 < 0:
                logging.warning(f"[AnalyzerEffects._combine_effects] {'for ' + marker if group_by_marker else ''} in {baseline} vs {pert}, Random effect model has negative tau2 (Between-study variance) — falling back to fixed effect.")
                effect_row = summary.loc[f"fixed effect"]
            
            if (effect_row["sd_eff"] is np.nan or np.isnan(effect_row['sd_eff'])) and effect_type=='random':
                logging.warning(f"[AnalyzerEffects._combine_effects] {'for ' + marker if group_by_marker else ''} in {baseline} vs {pert}, Random effect model has invalid variance estimate — falling back to fixed effect.")
                effect_row = summary.loc[f"fixed effect"]
            combined_effect = effect_row["eff"]
            combined_se = effect_row["sd_eff"]
            ci_low = effect_row["ci_low"]
            ci_upp = effect_row["ci_upp"]

            # Calc prediction interval if random effects
            if tau2 > 0:
                df = len(df) - 2 # degrees of freedom
                t_crit = t.ppf(0.975, df)
                pred_se = np.sqrt(combined_se**2 + tau2)

                lower_pi = combined_effect - t_crit * pred_se
                upper_pi = combined_effect + t_crit * pred_se
            else:
                lower_pi, upper_pi = np.nan, np.nan
            
            # Compute z and p: The pooled estimate is significantly different from zero.
            z = combined_effect / combined_se
            if alt == "two-sided":
                pvalue = 2 * (1 - norm.cdf(abs(z)))
            elif alt == "greater":
                pvalue = 1 - norm.cdf(z)
            elif alt == "smaller":
                pvalue = norm.cdf(z)

            p_heterogeneity  = 1 - chi2.cdf(meta_res.q, df=meta_res.df)
            I2 = max(0, (meta_res.q - meta_res.df) / meta_res.q) * 100

            tau2_hat, tau2_ci_low, tau2_ci_upp = self.__profiling_likelihood(effects, variances, tau2=tau2)

            # Check that tau2 from combine_effects matches profiling result
            tau_abs_tol = 1e-3
            if not math.isclose(tau2, tau2_hat, abs_tol=tau_abs_tol):
                logging.warning(f"tau2 from combine_effects ({tau2}) is different from tau2_hat from profiling ({tau2_hat}) for marker {marker} (abs_tol={tau_abs_tol})")
    
            result = {}
            if group_by_marker:
                result.update({'marker':marker}) 

            result.update({'baseline':baseline,'pert':pert,
                'combined_effect':combined_effect, 'combined_se':combined_se, 'pvalue':pvalue,
                'ci_low':ci_low, 'ci_upp':ci_upp, 'p_heterogeneity':p_heterogeneity, 
                'I2':I2, 'Q':meta_res.q, 'tau2':tau2, 'pi_low':lower_pi, 'pi_upp':upper_pi,
                'tau2_hat':tau2_hat, 'tau2_ci_low':tau2_ci_low, 'tau2_ci_upp':tau2_ci_upp})

            combined_effects.append(result)

        return pd.DataFrame(combined_effects)
    
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
        n_baseline_tiles = baseline_df.shape[0]
        n_pert_tiles = pert_df.shape[0]

        n_baseline_sites = baseline_df['site'].nunique()
        n_pert_sites = pert_df['site'].nunique()

        if min(n_baseline_sites, n_pert_sites) < min_required:
            logging.warning(f"Too few samples (sites): baseline_sites={n_baseline_sites}, pert_sites={n_pert_sites}. Minimum required sites is {min_required}")
            return {}
        logging.info(f"Sample size: baseline_tiles={n_baseline_tiles}, pert_tiles={n_pert_tiles}, baseline_sites={n_baseline_sites}, pert_sites={n_pert_sites}")

        effect_size, variance = self._compute_effect(baseline_df, pert_df, n_boot, embeddings_dim)

        return {'baseline_tiles_size':n_baseline_tiles, 'perturb_tiles_size':n_pert_tiles,
                'baseline_sites_size':n_baseline_sites, 'perturb_sites_size':n_pert_sites,
                 'effect_size': effect_size, 'variance':variance}

    def _prepare_embeddings_df(self, embeddings: np.ndarray[float], labels: np.ndarray[str], paths: np.ndarray[str]=None,
                            extract_marker:bool=True, extract_rep:bool=True) -> pd.DataFrame:
        """
        Create a DataFrame with embeddings and metadata parsed from sample labels.
        Parses labels of the form "marker_cellline_condition_batch_rep" into separate columns.

        Args:
            embeddings (np.ndarray[float]):     Embeddings array of shape (n_samples, n_features).
            labels (np.ndarray[str]):           Array of label strings matching embedding rows.
            paths (np.ndarray[str]):            Array of file paths corresponding to each sample.
            extract_marker (bool, Optional):    Whether to parse and include the 'marker' column from labels (default: True).
            extract_rep (bool, Optional):       Whether to parse and include the 'rep' column from labels (default: True).

        Returns:
            pd.DataFrame: DataFrame containing embedding columns plus columns:
                ['site', 'tile_index', 'marker', 'cell_line', 'condition', 'batch', 'rep'].

        Raises:
            ValueError: If any label string does not contain exactly 5 underscore-separated parts.
        """
        df = pd.DataFrame(embeddings)
        df['label'] = labels
        
        # Sagy 8.9.25
        if paths is not None:
            df['site'] = [p.split(os.sep)[-2] for p in paths]
            df['tile_index'] = [int(p.split(os.sep)[-1]) for p in paths]
        #

        # Split and validate
        split_labels = df['label'].str.split('_', expand=True)

        columns = ['cell_line', 'condition', 'batch']
        if extract_marker:
            columns.insert(0, 'marker')
        if extract_rep:
            columns.append('rep')

        # Check that all labels have len(columns) parts
        if split_labels.shape[1] != len(columns):
            invalid_labels = df['label'][split_labels.isnull().any(axis=1)].tolist()
            raise ValueError(
                f"Some label strings are invalid (expected {len(columns)} parts separated by '_').\n"
                f"Example invalid labels: {invalid_labels[:5]}"
            )
        df[columns] = split_labels
        return df

    
    def _calculate_all_effects(self, embeddings_df: pd.DataFrame, 
                               embeddings_dim:int, n_boot:int=1000) -> pd.DataFrame:
        """
        Calculate batch-level effect sizes for all marker-baseline-perturbation combinations.

        Args:
            embeddings_df (pd.DataFrame):
                DataFrame with embedding vectors and metadata, created by `_prepare_embeddings_df`.
            embeddings_dim (int):
                Dimensionality of the embedding vectors.
            n_boot (int)
                Number of bootstrap iterations (default: 1000).

        Returns:
            pd.DataFrame
                DataFrame with per-batch effect size statistics including:
                ['marker', 'baseline', 'pert', 'batch', 'baseline_size', 'perturb_size', 
                'effect_size', 'variance'].
        """

        results = []
        baseline = get_if_exists(self.data_config, 'BASELINE', None)
        assert baseline is not None, "BASELINE is None. You have to specify the baseline (for example: WT_Untreated or TDP43_Untreated)"

        logging.info(f"[AnalyzerEffects] baseline: {baseline}")
        baseline_cell_line, baseline_cond = baseline.split('_')
        baseline_df = embeddings_df[
                            (embeddings_df.cell_line == baseline_cell_line) &
                            (embeddings_df.condition == baseline_cond)]

        pert = get_if_exists(self.data_config, 'PERTURBATION', None)
        assert baseline is not None, "PERTURBATION is None. You have to specify the PERTURBATION (for example: WT_stress or TDP43_DOX)"

        logging.info(f"[AnalyzerEffects] pert: {pert}")
        pert_cell_line, pert_cond = pert.split('_')
        pert_df = embeddings_df[
                        (embeddings_df.cell_line == pert_cell_line) &
                        (embeddings_df.condition == pert_cond)]
        
        # Group each DataFrame by marker and batch separately
        baseline_groups = baseline_df.groupby(['marker', 'batch'])
        pert_groups = pert_df.groupby(['marker', 'batch'])
        # Iterate over marker-batch keys that appear in both baseline and perturbed
        common_batch_marker_keys = set(baseline_groups.groups.keys()) & set(pert_groups.groups.keys())
        common_batch_marker_keys = sorted(common_batch_marker_keys)
        for key in common_batch_marker_keys:
            marker, batch = key
            logging.info(f"[AnalyzerEffects] marker: {marker}, batch: {batch}")
            
            batch_marker_baseline_df = baseline_groups.get_group(key)
            batch_marker_pert_df = pert_groups.get_group(key)

            min_required = self.data_config.MIN_REQUIRED
            res = self._calculate_effect_per_unit(batch_marker_baseline_df, 
                                                    batch_marker_pert_df, embeddings_dim, 
                                                    min_required, n_boot)
            if res:
                res.update({'marker': marker, 'baseline': baseline, 'pert': pert, 'batch': batch})
                results.append(res)

        return pd.DataFrame(results)

    def __profiling_likelihood(self, effects:np.ndarray[float], variances:np.ndarray[float], tau2:float=None, tau2_max_expand:float=1e8):
        """
        Profile the restricted log-likelihood to estimate tau^2 (between-study variance)
        and its 95% confidence interval.

        Assuming tau2 was calculated using REML.
        
        Args:
            effects (np.ndarray[float]): Array of effect sizes from different studies/batches.
            variances (np.ndarray[float]): Array of variances corresponding to the effect sizes.
            tau2 (float, Optional): If provided, use this tau^2 value instead of estimating it with REML.
            tau2_max_expand (float, Optional): Maximum value to expand tau^2 search for CI (default: 1e8).

        Returns:
            Tuple[float, float, float]:
                - tau2_hat: Estimated tau^2 that maximizes the likelihood.
                - ci_low: Lower bound of the 95% confidence interval for tau^2.
                - ci_high: Upper bound of the 95% confidence interval for tau^2.
        
        """
        def __get_log_likelihood_reml(effects, variances, tau2):
            tau2 = max(tau2, 0.0)
            w = 1.0 / (variances + tau2)
            sw = np.sum(w)
            combined_effect = np.sum(w * effects) / sw
            Q = np.sum(w * (effects - combined_effect) ** 2)

            return -0.5 * (np.log(variances + tau2).sum() + np.log(sw) + Q)

        nll = lambda t2: -__get_log_likelihood_reml(effects, variances, t2) # negative log-likelihood

        if tau2 is None:
            logging.info(f"[__profiling_likelihood] Finding tau2_hat by minimizing negative log-likelihood (REML)")
            opt = optimize.minimize_scalar(nll, bounds=(0, 10), method="bounded")
            tau2_hat = opt.x  # the tau^2 that maximizes the likelihood
            ll_hat = -opt.fun # the maximum log-likelihood value (negative since we minimized minus (-) reml)
            logging.info(f"[__profiling_likelihood] tau2_hat={tau2_hat}, ll_hat={ll_hat}, success={opt.success}, message={opt.message}")
        else:
            tau2_hat = tau2
            ll_hat = __get_log_likelihood_reml(effects, variances, tau2_hat)
            logging.info(f"[__profiling_likelihood] Using provided tau2={tau2}: ll_hat={ll_hat}")

        def __ll_gap(tau2):
            # if gap < 0: tau2 is still inside the CI
            # if gap > 0: tau2 is outside the CI

            threshold = stats.chi2.ppf(1 - 0.05, df=1)
            gap = 2.0 * (ll_hat - __get_log_likelihood_reml(effects, variances, tau2)) - threshold

            return gap 

        # Find lower CI bound #
        if __ll_gap(0.0) <= 0.0:
            # optimize.brentq must have different signs for the endpoints of the range (ll_gap(0.0) and ll_gap(tau2_hat)), otherwise exception. 
            # If 0 is inside CI (i.e. __ll_gap(0.0) <= 0.0), like tau2_hat, then no sign change for the endpoints  brentq, so we set ci_low = 0.0
            ci_low = 0.0
        elif tau2_hat == 0:
            ci_low = 0.0 # if 0 is inside the CI, then the lower bound is 0
        else:
            ci_low = optimize.brentq(__ll_gap, 0.0, tau2_hat) # finds a number between 0 and tau2_hat such that ll_gap(b) = 0

        # Find upper CI bound #

        # Find the upper range to search for the CI by expanding until we find a point outside the CI
        tau2_high = max(tau2_hat, 1e-12)
        while __ll_gap(tau2_high) < 0 and tau2_high < tau2_max_expand:
            tau2_high *= 2.0 # expand CI until we find a point outside the CI

        # If we reached the maximum expansion limit without finding a point outside the CI return NaN
        if __ll_gap(tau2_high) < 0:
            logging.warning(f"[__profiling_likelihood] Could not find the upper CI bound. tau2_hat={tau2_hat}, tau2_high={tau2_high}, ll_gap(tau2_high)={__ll_gap(tau2_high)}")
            return tau2_hat, ci_low, np.nan

        # Search for the upper CI bound in the range [tau2_hat, tau2_high]
        ci_upper = optimize.brentq(__ll_gap, tau2_hat, tau2_high) # finds a number between tau2_hat and tau2_high such that ll_gap(b) = 0

        return tau2_hat, ci_low, ci_upper
