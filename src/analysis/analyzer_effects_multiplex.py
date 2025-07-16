import sys
import os
sys.path.insert(1, os.getenv("NOVA_HOME"))

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
import statsmodels.stats.meta_analysis as smm
from scipy.stats import norm, chi2

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

    def calculate(self, embeddings:np.ndarray[float], labels:np.ndarray[str], n_boot:int=1000)->Tuple[pd.DataFrame, pd.DataFrame]:
        
        baseline_perturb_dict = self._get_baseline_perturb_dict()
        output_folder_path = self.get_saving_folder(feature_type='effects_multiplex')
        logging.info(f'output folder path: {output_folder_path}')
        embeddings_df = self._prepare_embeddings_df(embeddings, labels)
        embeddings_dim = embeddings.shape[1]
        effects_df = self._calculate_all_effects(embeddings_df, baseline_perturb_dict, embeddings_dim, n_boot)
        combined_effects_df = self._combine_effects(effects_df)

        self.features = combined_effects_df, effects_df

        return combined_effects_df, effects_df

    
    def load(self)->None:
        """
        Load pre-calculated effects from a file into the self.features attribute.
        """
        output_folder_path = self.get_saving_folder(feature_type='effects_multiplex')
        loadpath_combined, loadpath = self._get_save_path(output_folder_path)
        self.features = pd.read_csv(loadpath_combined), pd.read_csv(loadpath)
        return None

    def save(self)->None:
        """
        Save the calculated effects to a specified file.
        """
        output_folder_path = self.get_saving_folder(feature_type='effects_multiplex')
        os.makedirs(output_folder_path, exist_ok=True)
        savepath_combined, savepath = self._get_save_path(output_folder_path)
        
        logging.info(f"Saving combined effects to {savepath_combined}")
        self.features[0].to_csv(savepath_combined, index=False)
        
        logging.info(f"Saving effects to {savepath}")
        self.features[1].to_csv(savepath, index=False)
        
        return None
    
    def _prepare_embeddings_df(self, embeddings: np.ndarray[float], 
                           labels: np.ndarray[str]) -> pd.DataFrame:
        """
        Create a DataFrame with embeddings and metadata parsed from sample labels.
        Parses labels of the form "marker_cellline_condition_batch_rep" into separate columns.

        Args:
            embeddings (np.ndarray[float]):     Embeddings array of shape (n_samples, n_features).
            labels (np.ndarray[str]):           Array of label strings matching embedding rows.

        Returns:
            pd.DataFrame: DataFrame containing embedding columns plus columns:
                ['cell_line', 'condition', 'batch'].

        Raises:
            ValueError: If any label string does not contain exactly 5 underscore-separated parts.
        """
        df = pd.DataFrame(embeddings)
        df['label'] = labels

        # Split and validate
        split_labels = df['label'].str.split('_', expand=True)

        # Check that all labels have 3 parts
        if split_labels.shape[1] != 3:
            invalid_labels = df['label'][split_labels.isnull().any(axis=1)].tolist()
            raise ValueError(
                f"Some label strings are invalid (expected 3 parts separated by '_').\n"
                f"Example invalid labels: {invalid_labels[:5]}"
            )
        df[['cell_line', 'condition', 'batch']] = split_labels
        return df

    
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
                    res = self._calculate_effect_per_batch(batch_baseline_df, 
                                                           batch_pert_df, embeddings_dim,
                                                           min_required, n_boot)
                    if res:
                        res.update({'batch': batch, 'baseline': baseline, 'pert': pert})
                        results.append(res)

        return pd.DataFrame(results)
    
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
        saveroot = os.path.join(feature_folder_path,f'{title}')
        return saveroot
    
    def _get_baseline_perturb_dict(self) -> Dict[str, list[str]]:
        """
        Retrieve baseline-to-perturbation mapping from the data configuration.
        """
        baseline_dict = get_if_exists(self.data_config, 'BASELINE_PERTURB', None)
        assert baseline_dict is not None, "BASELINE_PERTURB dict in data config is None. Example: {'WT_Untreated': ['WT_stress']}"
        return baseline_dict
    
    def _get_save_path(self, output_folder_path:str)->str:
        savepath_combined = os.path.join(output_folder_path, f"combined_effects.csv")
        savepath_batch = os.path.join(output_folder_path, f"effects.csv")
        return savepath_combined, savepath_batch 
    
    def _combine_effects(self, batch_effects: pd.DataFrame, alt: str = "greater", 
                     effect_type: str = 'random') -> pd.DataFrame:
        """
        Combine per-batch effect sizes into a single summary statistic per 
        cell_line using meta-analysis.
        Uses statsmodels' `combine_effects` with random effects by default.
        Falls back to fixed effects if random effects variance is invalid.

        Args:
            batch_effects (pd.DataFrame):   DataFrame containing per-batch effect sizes and 
                                            variances, with columns:
                                            ['baseline', 'pert', 'effect_size', 'variance', 'batch'].
            alt (str):                      Statistical alternative hypothesis for p-value 
                                            calculation. One of {'two-sided', 'greater', 'smaller'}.
            effect_type (str):              Meta-analysis model type: 'random' (default) or 'fixed'.

        Returns:
            pd.DataFrame:                   DataFrame with combined effect statistics per marker, including columns:
                                            ['marker', 'baseline', 'pert', 'combined_effect', 
                                            'combined_se', 'pvalue', 'ci_low', 'ci_upp', 
                                            'p_heterogeneity', 'I2', 'Q'].
        """
        if alt not in {"two-sided", "greater", "smaller"}:
            raise ValueError("Parameter 'alt' must be one of: 'two-sided', 'greater', or 'smaller'")
        if effect_type not in {"random", "fixed"}:
            raise ValueError("Parameter 'effect_type' must be one of: 'random', 'fixed'")
        
        combined_effects = []
        for (baseline, pert), cur_df_df in batch_effects.groupby(['baseline','pert']):
            effects = cur_df_df['effect_size'].values
            variances = cur_df_df['variance'].values
            
            # Run random effects meta-analysis
            meta_res = smm.combine_effects(effects, variances) # method_re=self.data_config.RE_METHOD
            
            summary = meta_res.summary_frame()
            tau2 = meta_res.tau2
            effect_row = summary.loc[f"{effect_type} effect"]
            if effect_type == 'random' and tau2 < 0:
                logging.warning(f"[AnalyzerEffects._combine_effects] in {baseline} vs {pert}, Random effect model has negative tau2 (Between-study variance) — falling back to fixed effect.")
                effect_row = summary.loc[f"fixed effect"]
            
            if (effect_row["sd_eff"] is np.nan or np.isnan(effect_row['sd_eff'])) and effect_type=='random':
                logging.warning(f"[AnalyzerEffects._combine_effects] in {baseline} vs {pert}, Random effect model has invalid variance estimate — falling back to fixed effect.")
                effect_row = summary.loc[f"fixed effect"]
            combined_effect = effect_row["eff"]
            combined_se = effect_row["sd_eff"]
            ci_low = effect_row["ci_low"]
            ci_upp = effect_row["ci_upp"]

            from scipy.stats import t
            k = len(cur_df_df)
            df = k-2
            t_crit = t.ppf(0.975, df)
            pred_se = np.sqrt(combined_se**2 + tau2)

            lower_pi = combined_effect - t_crit * pred_se
            upper_pi = combined_effect + t_crit * pred_se
            
            # Compute z and p
            z = combined_effect / combined_se
            if alt == "two-sided":
                pvalue = 2 * (1 - norm.cdf(abs(z)))
            elif alt == "greater":
                pvalue = 1 - norm.cdf(z)
            elif alt == "smaller":
                pvalue = norm.cdf(z)

            p_heterogeneity  = 1 - chi2.cdf(meta_res.q, df=meta_res.df)
            I2 = max(0, (meta_res.q - meta_res.df) / meta_res.q) * 100
    
            combined_effects.append({'baseline':baseline,'pert':pert,
                'combined_effect':combined_effect, 'combined_se':combined_se, 'pvalue':pvalue,
                'ci_low':ci_low, 'ci_upp':ci_upp, 'p_heterogeneity':p_heterogeneity, 
                'I2':I2, 'Q':meta_res.q, 'tau2':tau2, 'pi_low':lower_pi, 'pi_upp':upper_pi})
        return pd.DataFrame(combined_effects)