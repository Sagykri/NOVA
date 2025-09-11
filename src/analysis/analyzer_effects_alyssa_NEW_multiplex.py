import sys
import os
sys.path.insert(1, os.getenv("NOVA_HOME"))

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
import statsmodels.stats.meta_analysis as smm
from scipy.stats import norm, chi2

from src.analysis.analyzer_effects_multiplex import AnalyzerEffectsMultiplex
from src.datasets.dataset_config import DatasetConfig

from src.common.utils import get_if_exists
from src.datasets.label_utils import get_batches_from_input_folders

patient_to_plate = {'EDi022':1,'EDi029':2,'EDi037':3}
class AnalyzerEffectsAlyssaNEWMultiplex(AnalyzerEffectsMultiplex):
    
    def __init__(self, data_config: DatasetConfig, output_folder_path:str):
        """Get an instance

        Args:
            data_config (DatasetConfig): The dataset configuration object. 
            output_folder_path (str): path to output folder
        """
        super().__init__(data_config, output_folder_path)
    
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
        # Flat the baseline-perturbation dictionary to get all unique cell lines and conditions, remove the patient ID
        cell_lines_conditions = [f"{cl.split('-')[0]}_{cond}" for cl, cond in (bc.split('_') for bc in baseline_pertrub_dict.keys())]
        cell_lines_conditions = np.unique(cell_lines_conditions).tolist()

        title = f"{'_'.join(input_folders)}_{'_'.join(cell_lines_conditions)}"
        saveroot = os.path.join(feature_folder_path,f'{title}')
        
        return saveroot

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
                        baseline_general, baseline_patient = baseline_cell_line.split('-')
                        pert_general, pert_patient = pert_cell_line.split('-')
                        res.update({'batch': batch, 'baseline': baseline_general, 'pert': pert_general,
                                    'baseline_patient':baseline_patient, 'pert_patient':pert_patient,
                                    'plate':patient_to_plate[baseline_patient]})
                        results.append(res)

        return pd.DataFrame(results)