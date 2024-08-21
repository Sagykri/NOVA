from abc import abstractmethod

from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig ## TODO SAGY CHANGE

from src.Analysis.analyzer import Analyzer
from src.common.lib.utils import handle_log, get_if_exists

import logging
import os
import pandas as pd


class AnalyzerDistances(Analyzer):
    def __init__(self, trainer_conf: TrainerConfig, data_conf: DatasetConfig):
        super().__init__(trainer_conf, data_conf)


    @abstractmethod
    def calculate(self, embeddings, labels):
        pass
    
    def load(self):
        model_output_folder = self.output_folder_path
        handle_log(model_output_folder)
        logging.info(f"[load scores]: model_output_folder: {model_output_folder}")
        
        output_folder_path = os.path.join(model_output_folder, 'figures', self.experiment_type, 'distances')
        if not os.path.exists(output_folder_path):
            logging.info(f"[load scores]: model_output_folder: {model_output_folder} does not exists, can't load!")
        
        batches = [input_folder.split(os.sep)[-1] for input_folder in self.input_folders] 
        baseline_cell_line_cond = get_if_exists(self.data_conf, 'BASELINE_CELL_LINE_CONDITION', None)

        savepath = os.path.join(output_folder_path, f"metrics_score{'_'.join(batches)}_{baseline_cell_line_cond}.csv")
        self.features = pd.read_csv(savepath)
        return None

    def save(self):
        model_output_folder = self.output_folder_path
        handle_log(model_output_folder)
        logging.info(f"[save scores]: model_output_folder: {model_output_folder}")
        
        output_folder_path = os.path.join(model_output_folder, 'figures', self.experiment_type, 'distances')
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path, exist_ok=True)
        
        batches = [input_folder.split(os.sep)[-1] for input_folder in self.input_folders] 
        baseline_cell_line_cond = get_if_exists(self.data_conf, 'BASELINE_CELL_LINE_CONDITION', None)

        savepath = os.path.join(output_folder_path, f"metrics_score{'_'.join(batches)}_{baseline_cell_line_cond}.csv")
        logging.info(f"Saving scores to {savepath}")
        self.features.to_csv(savepath, index=False)
        return None
    
    