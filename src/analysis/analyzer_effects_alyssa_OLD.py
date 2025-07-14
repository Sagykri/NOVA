import sys
import os
sys.path.insert(1, os.getenv("NOVA_HOME"))

from abc import abstractmethod
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
from src.analysis.analyzer_effects_dist_ratio import AnalyzerEffectsDistRatio
from src.datasets.dataset_config import DatasetConfig

from src.common.utils import get_if_exists
from src.datasets.label_utils import get_batches_from_input_folders

class AnalyzerEffectsAlyssaOLD(AnalyzerEffectsDistRatio):
    
    def __init__(self, data_config: DatasetConfig, output_folder_path:str):
        """Get an instance

        Args:
            data_config (DatasetConfig): The dataset configuration object. 
            output_folder_path (str): path to output folder
        """
        super().__init__(data_config, output_folder_path)

    def calculate(self, embeddings:np.ndarray[float], labels:np.ndarray[str], n_boot:int=1000)->Tuple[pd.DataFrame, pd.DataFrame]:
        
        baseline_perturb_dict = self._get_baseline_perturb_dict()
        output_folder_path = self.get_saving_folder(feature_type='effects')
        logging.info(f'output folder path: {output_folder_path}')
        embeddings_df = self._prepare_embeddings_df(embeddings, labels)
        embeddings_dim = embeddings.shape[1]
        effects_df = self._calculate_all_effects(embeddings_df, baseline_perturb_dict, embeddings_dim, n_boot)
        self.features = effects_df
        return effects_df
    
    def load(self)->None:
        """
        Load pre-calculated effects from a file into the self.features attribute.
        """
        output_folder_path = self.get_saving_folder(feature_type='effects')
        loadpath = self._get_save_path(output_folder_path)
        self.features = pd.read_csv(loadpath)
        return None

    def save(self)->None:
        """
        Save the calculated effects to a specified file.
        """
        output_folder_path = self.get_saving_folder(feature_type='effects')
        os.makedirs(output_folder_path, exist_ok=True)
        savepath = self._get_save_path(output_folder_path)
        
        logging.info(f"Saving batch effects to {savepath}")
        self.features.to_csv(savepath, index=False)
        
        return None
    
    def _calculate_effect_per_marker(self, marker_baseline_df:pd.DataFrame, 
                                    marker_pert_df:pd.DataFrame, embeddings_dim: int, 
                                    min_required:int, n_boot:int=1000):
        
        baseline_embeddings = marker_baseline_df.iloc[:, :embeddings_dim].values

        perturb_embeddings = marker_pert_df.iloc[:, 0:embeddings_dim].values

        n_baseline = baseline_embeddings.shape[0]
        n_pert = perturb_embeddings.shape[0]
        if min(n_baseline, n_pert) < min_required:
            logging.warning(f"Too few samples: baseline={n_baseline}, pert={n_pert}. Minimum required is {min_required}")
            return {}
        
        effect_size, pval = self._compute_effect_and_pval(baseline_embeddings, perturb_embeddings, n_boot)

        return {'baseline_size':n_baseline, 'perturb_size':n_pert,
                 'effect_size': effect_size, 'pvalue':pval}

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
                ['marker', 'cell_line', 'condition', 'batch', 'rep'].

        Raises:
            ValueError: If any label string does not contain exactly 5 underscore-separated parts.
        """
        df = pd.DataFrame(embeddings)
        df['label'] = labels

        # Split and validate
        split_labels = df['label'].str.split('_', expand=True)

        # Check that all labels have 5 parts
        if split_labels.shape[1] != 5:
            invalid_labels = df['label'][split_labels.isnull().any(axis=1)].tolist()
            raise ValueError(
                f"Some label strings are invalid (expected 5 parts separated by '_').\n"
                f"Example invalid labels: {invalid_labels[:5]}"
            )
        df[['marker', 'cell_line', 'condition', 'batch', 'rep']] = split_labels
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
                
                # Group each DataFrame by marker separately
                baseline_groups = baseline_df.groupby(['marker'])
                pert_groups = pert_df.groupby(['marker'])
                # Iterate over marker keys that appear in both baseline and perturbed
                common_marker_keys = set(baseline_groups.groups.keys()) & set(pert_groups.groups.keys())
                common_marker_keys = sorted(common_marker_keys)
                for marker in common_marker_keys:
                    logging.info(f"[AnalyzerEffects] marker: {marker}")
                    
                    marker_baseline_df = baseline_groups.get_group(marker)
                    marker_pert_df = pert_groups.get_group(marker)

                    min_required = self.data_config.MIN_REQUIRED
                    res = self._calculate_effect_per_marker(marker_baseline_df, 
                                                           marker_pert_df, embeddings_dim,
                                                           min_required, 
                                                           n_boot)
                    if res:
                        res.update({'marker': marker, 'baseline': baseline, 'pert': pert})
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
    
    @abstractmethod    
    def _compute_effect(self, group_baseline: np.ndarray[float], group_pert: np.ndarray[float])->float:
        """
        Computes Log2FC effect size between two groups based on distances to the baseline medoid.

        Computes the medoid of the baseline group, calculates distances of 
        both baseline and perturbed samples to that medoid, and then computes the ratio between 
        their median distances. Eventually, apply Log2 on the ratio.

        Args:
            group_baseline (np.ndarray): Array of shape (n_samples_baseline, n_features),
            group_pert (np.ndarray): Array of shape (n_samples_pert, n_features),

        Returns:
            np.ndarray: float (effect size)
        """
        centroid_baseline = np.median(group_baseline,axis=0, keepdims=True)
        dists_baseline = np.linalg.norm(group_baseline - centroid_baseline, axis=1)
        dists_pert = np.linalg.norm(group_pert - centroid_baseline, axis=1)

        median_dist_baseline = np.median(dists_baseline)
        median_dist_pert = np.median(dists_pert)
        
        effect_size = np.log2(median_dist_pert / median_dist_baseline)        
        return effect_size
    
    
    def _compute_effect_and_pval(self, group_baseline: np.ndarray[float], group_pert: np.ndarray[float],
                      n_perm: int = 1000, random_state: int = 0) -> float:
        """
        Compute effect size and p-value by permutation test: 
        shuffling group labels and recomputing effect.
        """
        rng = np.random.default_rng(random_state)

        combined = np.concatenate([group_baseline, group_pert], axis=0)
        labels = np.array([0] * len(group_baseline) + [1] * len(group_pert))
        
        observed = self._compute_effect(group_baseline, group_pert)

        count = 0
        for _ in range(n_perm):
            rng.shuffle(labels)
            perm_baseline = combined[labels == 0]
            perm_pert = combined[labels == 1]
            if len(perm_baseline) == 0 or len(perm_pert) == 0:
                continue  # Skip invalid splits
            perm_effect = self._compute_effect(perm_baseline, perm_pert)
            if perm_effect >= observed:
                count += 1
            
        # +1 smoothing correction:
        pval = (count + 1) / (n_perm + 1)
        return observed, pval
    
    def _get_save_path(self, output_folder_path:str)->str:
        savepath = os.path.join(output_folder_path, f"effects.csv")
        return savepath 