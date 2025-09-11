import sys
import os
sys.path.insert(1, os.getenv("NOVA_HOME"))

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
from src.analysis.analyzer_effects_dist_ratio import AnalyzerEffectsDistRatio
from src.datasets.dataset_config import DatasetConfig

from src.common.utils import get_if_exists
from src.datasets.label_utils import get_batches_from_input_folders

patient_to_plate = {'EDi022':1,'EDi029':2,'EDi037':3}
class AnalyzerEffectsAlyssaNEW(AnalyzerEffectsDistRatio):
    
    def __init__(self, data_config: DatasetConfig, output_folder_path:str):
        """Get an instance

        Args:
            data_config (DatasetConfig): The dataset configuration object. 
            output_folder_path (str): path to output folder
        """
        super().__init__(data_config, output_folder_path)

    def calculate(self, embeddings:np.ndarray[float], labels:np.ndarray[str], paths: np.ndarray[str], n_boot:int=1000)->Tuple[pd.DataFrame, pd.DataFrame]:
        
        baseline_perturb_dict = self._get_baseline_perturb_dict()
        output_folder_path = self.get_saving_folder(feature_type=self.feature_type)
        logging.info(f'output folder path: {output_folder_path}')
        embeddings_df = self._prepare_embeddings_df(embeddings, labels, paths)
        embeddings_dim = embeddings.shape[1]
        effects_df = self._calculate_all_effects(embeddings_df, baseline_perturb_dict, embeddings_dim, n_boot)
        combined_effects_df = self._combine_effects(effects_df)
        self._correct_for_multiple_hypothesis(combined_effects_df)

        self.features = combined_effects_df, effects_df

        return combined_effects_df, effects_df
    
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
                    res = self._calculate_effect_per_unit(marker_baseline_df, 
                                                           marker_pert_df, embeddings_dim,
                                                           min_required, n_boot)
                    if res:
                        baseline_general, baseline_patient = baseline_cell_line.split('-')
                        pert_general, pert_patient = pert_cell_line.split('-')
                        res.update({'marker': marker, 'baseline': baseline_general, 'pert': pert_general,
                                    'baseline_patient':baseline_patient, 'pert_patient':pert_patient,
                                    'plate':patient_to_plate[baseline_patient]})
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