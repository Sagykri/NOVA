
import sys
import os

from src.datasets.label_utils import get_batches_from_input_folders
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from abc import abstractmethod
import logging
import os
import pandas as pd
import numpy as np
from typing import List
import itertools
from sklearn.metrics.pairwise import pairwise_distances

from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig
from src.analysis.analyzer import Analyzer
from src.common.lib.utils import get_if_exists

class AnalyzerDistancesFull(Analyzer):
    def __init__(self, trainer_config: TrainerConfig, data_config: DatasetConfig):
        super().__init__(trainer_config, data_config)


    def calculate(self, embeddings:np.ndarray[float], labels:np.ndarray[str])->pd.DataFrame:
        """Calculate distance metrics from given embeddings, save in the self.features attribute

        Args:
            embeddings (np.ndarray[float]): The embeddings
            labels (np.ndarray[str]): The corresponding labels of the embeddings
        """
        # First we must define what is the baseline cell line condition of the calculation: 
        # we always calculate the distance for a given marker between one condition and the baseline condition.
        baseline_cell_line_cond = get_if_exists(self.data_config, 'BASELINE_CELL_LINE_CONDITION', None)
        assert baseline_cell_line_cond is not None, "BASELINE_CELL_LINE_CONDITION is None. You have to specify the baseline to calculate the distance score against (for example: WT_Untreated or TDP43_Untreated)"

        # Next we extract info from our labels: which other conditions we have, which unique markers and batches we can work with
        conditions = get_unique_cell_lines_conds_from_labels(labels)
        conditions = np.delete(conditions, np.where(conditions == baseline_cell_line_cond)[0]) # Remove the baseline
        markers = get_unique_markers_from_labels(labels)
        batches = get_unique_batches_from_labels(labels)

        logging.info(f"Conditions: {conditions}, markers: {markers}, batches: {batches}, baseline_cell_line_cond: {baseline_cell_line_cond}")

        scores = pd.DataFrame()

        for batch in batches:
            logging.info(f"batch: {batch}")
            print(batch)
            for marker in markers:  
                print(marker) 
                logging.info(f"marker: {marker}")
                scores = self._calculate_metrics_for_batch_and_marker(embeddings, labels, baseline_cell_line_cond,
                                                  conditions, batch, marker, scores)
                
            

        self.features = scores
        return scores
    
    def load(self)->None:
        """load the saved distances into the self.features attribute
        """
        model_output_folder = self.output_folder_path
        logging.info(f"[load scores]: model_output_folder: {model_output_folder}")
        
        output_folder_path = os.path.join(model_output_folder, 'figures', self.data_config.EXPERIMENT_TYPE, 'distances_full')
        if not os.path.exists(output_folder_path):
            logging.info(f"[load scores]: model_output_folder: {model_output_folder} does not exists, can't load!")
        
        batches = get_batches_from_input_folders(self.data_config.INPUT_FODLERS)
        baseline_cell_line_cond = get_if_exists(self.data_config, 'BASELINE_CELL_LINE_CONDITION', None)

        savepath = os.path.join(output_folder_path, f"metrics_score_full_distances_{'_'.join(batches)}_{baseline_cell_line_cond}.csv")
        self.features = pd.read_csv(savepath)
        return None

    def save(self):
        """save the calculated distances in path derived from self.output_folder_path
        """
        model_output_folder = self.output_folder_path
        logging.info(f"[save scores]: model_output_folder: {model_output_folder}")
        
        output_folder_path = os.path.join(model_output_folder, 'figures', self.data_config.EXPERIMENT_TYPE, 'distances_full')
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path, exist_ok=True)
        
        batches = get_batches_from_input_folders(self.data_config.INPUT_FOLDERS)
        baseline_cell_line_cond = get_if_exists(self.data_config, 'BASELINE_CELL_LINE_CONDITION', None)

        savepath = os.path.join(output_folder_path, f"metrics_score_full_distances_{'_'.join(batches)}_{baseline_cell_line_cond}.csv")
        logging.info(f"Saving scores to {savepath}")
        self.features.to_csv(savepath, index=False)
        return None

    def _calculate_metrics_for_batch_and_marker(self, embeddings:np.ndarray[float], labels:np.ndarray[str],
                                                baseline_cell_line_cond:str, conditions:List[str], batch:str,
                                                marker:str, scores:pd.DataFrame)->pd.DataFrame:
        """Protected method to calculate the wanted distance metric for a given batch and marker.

        Args:
            embeddings (np.ndarray[float]): all embeddings to calculate the distance for
            labels (np.ndarray[str]): corresponding labels of the embeddings
            baseline_cell_line_cond (str): the 'cell_line_condition' that will be the baseline of the distances
            conditions (List[str]): a list of all the 'cell_line_condition' to compare to baseline
            batch (str): batch number to calculate the distance for
            marker (str): marker to calculate the distance for
            scores (pd.DataFrame): dataframe to store the results in

        Returns:
            pd.DataFrame: updated dataframe with the results
        """
        

        scores = self._calculate_metric_for_baseline(embeddings, labels, baseline_cell_line_cond, batch, marker, scores)
        # Then we calculate the difference score between the conditions and the baseline:
        for cond in conditions:
            logging.info(f"cell line: {cond}")
            print(cond)
            scores = self._calculate_metric_for_cond(embeddings, labels, baseline_cell_line_cond, cond, batch, marker, scores)
        return scores
    
       
    
    def _calculate_metric_for_cond(self, embeddings:np.ndarray[float], labels:np.ndarray[str],
                                    baseline_cell_line_cond:str, cond:str,batch:str,
                                    marker:str, scores:pd.DataFrame)->pd.DataFrame:
        """Protected method to calculate the wanted distance metric between the baseline condition and a given condition
        Args:
            embeddings (np.ndarray[float]): all embeddings to calculate the distance for
            labels (np.ndarray[str]): corresponding labels of the embeddings
            baseline_cell_line_cond (str): the 'cell_line_condition' that will be the baseline of the distances
            cond (str): the condition to measure the distance from baseline
            batch (str): batch number to calculate the distance for
            marker (str): marker to calculate the distance for
            scores (pd.DataFrame): dataframe to store the results in

        Returns:
            pd.DataFrame: updated dataframe with the results
        """
        reps = itertools.product(['rep1', 'rep2'], repeat=2)
        for repA,repB in reps:
            print(repA,repB)
            condition_indices = np.where(np.char.find(labels.astype(str), f'{marker}_{cond}_{batch}_{repA}')>-1)[0]
            baseline_indices = np.where(np.char.find(labels.astype(str), f'{marker}_{baseline_cell_line_cond}_{batch}_{repB}')>-1)[0]
            
            logging.info(f"{marker}_{cond}_{batch}_{repA} size: {len(condition_indices)}")
            logging.info(f"{marker}_{baseline_cell_line_cond}_{batch}_{repB} size: {len(baseline_indices)}")

            if len(condition_indices) == 0:
                logging.info(f"No samples for {marker}_{cond}_{batch}_{repA}. Skipping...")
                continue
            if len(baseline_indices) == 0:
                logging.info(f"No samples for {marker}_{baseline_cell_line_cond}_{repB}. Skipping...")
                continue
            
            distances = pairwise_distances(embeddings[baseline_indices],embeddings[condition_indices] )
            score = pd.DataFrame(data={'dist':distances.flatten(),'marker':marker,'condition':cond,'repA':repA,'repB':repB,'batch':batch})

            if scores.shape[0]==0:
                scores = score
            else:
                scores = pd.concat([scores, score], ignore_index=True)

        return scores
    
    
    def _calculate_metric_for_baseline(self, embeddings:np.ndarray[float], labels:np.ndarray[str],  
                                       baseline_cell_line_cond:str, batch:str,
                                       marker:str, scores:pd.DataFrame)->pd.DataFrame:
        """Protected method to calculate the wanted distance metric between the baseline condition and a given condition
        Args:
            embeddings (np.ndarray[float]): all embeddings to calculate the distance for
            labels (np.ndarray[str]): corresponding labels of the embeddings
            baseline_cell_line_cond (str): the 'cell_line_condition' that will be the baseline of the distances
            cond (str): the condition to measure the distance from baseline
            batch (str): batch number to calculate the distance for
            marker (str): marker to calculate the distance for
            scores (pd.DataFrame): dataframe to store the results in

        Returns:
            pd.DataFrame: updated dataframe with the results
        """
        repA = 'rep1'
        repB = 'rep2'
        baseline_rep1_indices = np.where(np.char.find(labels.astype(str), f'{marker}_{baseline_cell_line_cond}_{batch}_{repA}')>-1)[0]
        baseline_rep2_indices = np.where(np.char.find(labels.astype(str), f'{marker}_{baseline_cell_line_cond}_{batch}_{repB}')>-1)[0]
        
        logging.info(f"{marker}_{baseline_cell_line_cond}_{batch}_{repA} size: {len(baseline_rep1_indices)}")
        logging.info(f"{marker}_{baseline_cell_line_cond}_{batch}_{repB} size: {len(baseline_rep2_indices)}")

        if len(baseline_rep1_indices) == 0:
            logging.info(f"No samples for {marker}_{baseline_cell_line_cond}_{batch}_{repA}. Skipping...")
            return scores
        if len(baseline_rep2_indices) == 0:
            logging.info(f"No samples for {marker}_{baseline_cell_line_cond}_{repB}. Skipping...")
            return scores
        
        distances = pairwise_distances(embeddings[baseline_rep1_indices],embeddings[baseline_rep2_indices] )
        score = pd.DataFrame(data={'dist':distances.flatten(),'marker':marker,'condition':baseline_cell_line_cond,'repA':repA,'repB':repB,'batch':batch})

        if scores.shape[0]==0:
            scores = score
        else:
            scores = pd.concat([scores, score], ignore_index=True)

        return scores