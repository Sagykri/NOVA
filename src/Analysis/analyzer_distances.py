
from abc import abstractmethod
import logging
import os
import pandas as pd
import numpy as np
import datetime
from typing import List
import itertools

from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig ## TODO SAGY CHANGE
from src.Analysis.analyzer import Analyzer
from src.common.lib.utils import get_if_exists

class AnalyzerDistances(Analyzer):
    def __init__(self, trainer_conf: TrainerConfig, data_conf: DatasetConfig):
        super().__init__(trainer_conf, data_conf)


    def calculate(self, embeddings:np.ndarray, labels:np.ndarray)->None:
        """Calculate distance metrics from given embeddings, save in the self.features attribute

        Args:
            embeddings (np.ndarray): The embeddings
            labels (np.ndarray): The corresponding labels of the embeddings
        """
        baseline_cell_line_cond = get_if_exists(self.data_conf, 'BASELINE_CELL_LINE_CONDITION', None)
        assert baseline_cell_line_cond is not None, "BASELINE_CELL_LINE_CONDITION is None. You have to specify the baseline to calculate the silhouette against (for example: WT_Untreated or TDP43_Untreated)"

        conditions = np.unique(['_'.join(l.split("_")[1:3]) for l in labels]) # Extract the list of cell_line_conds
        conditions = np.delete(conditions, np.where(conditions == baseline_cell_line_cond)[0]) # Remove the baseline
        markers = np.unique([label.split('_')[0] for label in np.unique(labels)])
        batches = [input_folder.split(os.sep)[-1] for input_folder in self.input_folders] 

        logging.info(f"Conditions: {conditions}, markers: {markers}, batches: {batches}, baseline_cell_line_cond: {baseline_cell_line_cond}")

        scores = pd.DataFrame()

        for batch in batches:
            logging.info(f"batch: {batch}")
            for marker in markers:   
                logging.info(f"marker: {marker}")
                scores = self._calculate_metrics_for_batch_and_marker(embeddings, labels, baseline_cell_line_cond,
                                                  conditions, batch, marker, scores)
            

        self.features = scores
        return None
    
    def load(self)->None:
        """load the saved distances into the self.features attribute
        """
        model_output_folder = self.output_folder_path
        logging.info(f"[load scores]: model_output_folder: {model_output_folder}")
        
        output_folder_path = os.path.join(model_output_folder, 'figures', self.experiment_type, 'distances')
        if not os.path.exists(output_folder_path):
            logging.info(f"[load scores]: model_output_folder: {model_output_folder} does not exists, can't load!")
        
        batches = [input_folder.split(os.sep)[-1] for input_folder in self.input_folders] 
        baseline_cell_line_cond = get_if_exists(self.data_conf, 'BASELINE_CELL_LINE_CONDITION', None)

        savepath = os.path.join(output_folder_path, f"metrics_score_{'_'.join(batches)}_{baseline_cell_line_cond}.csv")
        self.features = pd.read_csv(savepath)
        return None

    def save(self):
        """save the calculated distances in path derived from self.output_folder_path
        """
        model_output_folder = self.output_folder_path
        logging.info(f"[save scores]: model_output_folder: {model_output_folder}")
        
        output_folder_path = os.path.join(model_output_folder, 'figures', self.experiment_type, 'distances')
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path, exist_ok=True)
        
        batches = [input_folder.split(os.sep)[-1] for input_folder in self.input_folders] 
        baseline_cell_line_cond = get_if_exists(self.data_conf, 'BASELINE_CELL_LINE_CONDITION', None)

        __now = datetime.datetime.now()
        savepath = os.path.join(output_folder_path, f"metrics_score_{'_'.join(batches)}_{baseline_cell_line_cond}_{__now.strftime('%d%m%y_%H%M%S_%f')}.csv") #TODO: remove time!
        logging.info(f"Saving scores to {savepath}")
        self.features.to_csv(savepath, index=False)
        return None

    @abstractmethod    
    def _compute_scores(self, cur_embeddings: np.ndarray, cur_labels: np.ndarray) -> pd.DataFrame:
        """Abstract method to compute the actual score

        Args:
            cur_embeddings (np.ndarray): embeddings to calculate scores on
            cur_labels (np.ndarray): labels of the embeddings to calculate scores on; should contain only 2 unique labels

        Returns:
            pd.DataFrame: dataframe containing the score 
        """

    def _calculate_metrics_for_batch_and_marker(self, embeddings:np.ndarray, labels:np.ndarray,
                                                  baseline_cell_line_cond:str,
                                                  conditions:List[str], batch:str,
                                                  marker:str, scores:pd.DataFrame)->pd.DataFrame:
        """Protected method to calculate the wanted distance metric for a given batch and marker.

        Args:
            embeddings (np.ndarray): all embeddings to calculate the distance for
            labels (np.ndarray): corresponding labels of the embeddings
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
            scores = self._calculate_metric_for_condition(embeddings, labels, baseline_cell_line_cond, cond, batch, marker, scores)

        return scores
    
    def _calculate_metric_for_baseline(self, embeddings:np.ndarray, labels:np.ndarray,
                                        baseline_cell_line_cond:str,
                                        batch:str,
                                        marker:str, scores:pd.DataFrame)->pd.DataFrame:
        """Protected method to calculate the wanted distance metric between the baseline condition and itself,
        using random artificial splitting of each rep into two halves.

        Args:
            embeddings (np.ndarray): all embeddings to calculate the distance for
            labels (np.ndarray): corresponding labels of the embeddings
            baseline_cell_line_cond (str): the 'cell_line_condition' that will be the baseline of the distances
            batch (str): batch number to calculate the distance for
            marker (str): marker to calculate the distance for
            scores (pd.DataFrame): dataframe to store the results in

        Returns:
            pd.DataFrame: updated dataframe with the results
        """
        np.random.seed(1)
        # We start with calculating the score of difference between the baseline and itself
        logging.info(f"cell line (baseline): {baseline_cell_line_cond}")
        rep1_indices = np.where(np.char.find(labels.astype(str), f'{marker}_{baseline_cell_line_cond}_{batch}_rep1')>-1)[0]
        rep2_indices = np.where(np.char.find(labels.astype(str), f'{marker}_{baseline_cell_line_cond}_{batch}_rep2')>-1)[0]

        if len(rep1_indices) == 0 and len(rep2_indices) == 0:
            logging.warn(f"Marker {marker} couldn't be found in batch {batch}. Skipping this marker..")
            return scores

        r1_part1, r1_part2 = self._random_split_indices(rep1_indices)
        r2_part1, r2_part2 = self._random_split_indices(rep2_indices)

        logging.info(f"Baseline split sizes: r1_part1={len(r1_part1)}, r1_part2={len(r1_part2)}, r2_part1={len(r2_part1)}, r2_part2={len(r2_part2)}")

        partial_reps = itertools.product(['rep1_part1', 'rep1_part2'], ['rep2_part1', 'rep2_part2'])
        indices_dict = {'rep1_part1': r1_part1, 'rep1_part2': r1_part2, 'rep2_part1': r2_part1, 'rep2_part2': r2_part2}
   
        for repA, repB in partial_reps:
            repA_indices, repB_indices = indices_dict[repA], indices_dict[repB]
            cur_labels = np.concatenate([[repA]*len(repA_indices),[repB]*len(repB_indices)])
            cur_embeddings = np.concatenate([embeddings[repA_indices],embeddings[repB_indices]])
            
            score = self._compute_scores(cur_embeddings, cur_labels)
            score['marker'] = marker
            score['condition'] = baseline_cell_line_cond
            score['repA'] = repA
            score['repB'] = repB
            score['batch'] = batch

            if scores.shape[0]==0:
                scores = score
            else:
                scores = pd.concat([scores, score], ignore_index=True)
        return scores
    
    def _random_split_indices(self, indices:np.ndarray)->tuple[np.ndarray,np.ndarray]:
        """Randomly split indices into two parts.

        Args:
            indices (np.ndarray): indices to split

        Returns:
            tuple[np.ndarray,np.ndarray]: the two splitted indices array
        """
        half_size = len(indices) // 2
        part1 = np.random.choice(indices, size=half_size, replace=False)
        part2 = np.setdiff1d(indices, part1)
    
        return part1, part2
    
    def _calculate_metric_for_condition(self, embeddings:np.ndarray, labels:np.ndarray,
                                        baseline_cell_line_cond:str,
                                        cond:str,batch:str,
                                        marker:str, scores:pd.DataFrame)->pd.DataFrame:
        """Protected method to calculate the wanted distance metric between the baseline condition and a given condition
        Args:
            embeddings (np.ndarray): all embeddings to calculate the distance for
            labels (np.ndarray): corresponding labels of the embeddings
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
            
            cur_labels = np.concatenate([labels[baseline_indices],labels[condition_indices]])
            cur_embeddings = np.concatenate([embeddings[baseline_indices],embeddings[condition_indices]])
            
            score = self._compute_scores(cur_embeddings, cur_labels)
            score['marker'] = marker
            score['condition'] = cond
            score['repA'] = repA
            score['repB'] = repB
            score['batch'] = batch

            if scores.shape[0]==0:
                scores = score
            else:
                scores = pd.concat([scores, score], ignore_index=True)

        return scores
    
    