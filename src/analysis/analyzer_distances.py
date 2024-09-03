
import sys
import os
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from abc import abstractmethod
import logging
import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Iterable, Dict
import itertools

from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig
from src.analysis.analyzer import Analyzer
from src.common.lib.utils import get_if_exists, get_unique_cell_lines_conds_from_labels, get_unique_markers_from_labels, get_unique_batches_from_labels, get_batches_from_input_folders

class AnalyzerDistances(Analyzer):
    def __init__(self, trainer_config: TrainerConfig, data_config: DatasetConfig):
        super().__init__(trainer_config, data_config)


    def calculate(self, embeddings:np.ndarray[float], labels:np.ndarray[str])->pd.DataFrame:
        """Calculate distance metrics from given embeddings, save in the self.features attribute
        and return it as well

        Args:
            embeddings (np.ndarray[float]): The embeddings
            labels (np.ndarray[str]): The corresponding labels of the embeddings
        Returns:
            pd.DataFrame: DataFrame containing the calculated distances
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
        logging.info(f"[AnalyzerDistances.calculate] Conditions: {conditions}, markers: {markers}, batches: {batches}, baseline_cell_line_cond: {baseline_cell_line_cond}")

        # Finally we can calculate the distances, separatly for each batch and marker
        scores = pd.DataFrame()
        for batch in batches:
            logging.info(f"[AnalyzerDistances.calculate] batch: {batch}")
            for marker in markers:   
                logging.info(f"[AnalyzerDistances.calculate] marker: {marker}")
                new_scores = self._calculate_metrics_for_batch_and_marker(embeddings, labels, baseline_cell_line_cond,
                                                  conditions, batch, marker)
                scores = pd.concat([scores,new_scores], ignore_index=True)
            

        self.features = scores
        return scores
    
    def load(self)->None:
        """load the saved distances into the self.features attribute
        """
        output_folder_path = self._get_saving_folder() #TODO: adjust also analyzer umap to do the same
        logging.info(f"[save scores]: output_folder_path: {output_folder_path}")
        batches = get_batches_from_input_folders(self.data_config.INPUT_FODLERS)
        baseline_cell_line_cond = get_if_exists(self.data_config, 'BASELINE_CELL_LINE_CONDITION', None)

        loadpath = os.path.join(output_folder_path, f"metrics_score_{'_'.join(batches)}_{baseline_cell_line_cond}.csv")
        self.features = pd.read_csv(loadpath)
        return None

    def save(self):
        """save the calculated distances in path derived from self.output_folder_path
        """
        output_folder_path = self._get_saving_folder()
        logging.info(f"[save scores]: output_folder_path: {output_folder_path}")
        
        batches = get_batches_from_input_folders(self.data_config.INPUT_FODLERS)
        baseline_cell_line_cond = get_if_exists(self.data_config, 'BASELINE_CELL_LINE_CONDITION', None)

        savepath = os.path.join(output_folder_path, f"metrics_score_{'_'.join(batches)}_{baseline_cell_line_cond}.csv")
        logging.info(f"Saving scores to {savepath}")
        self.features.to_csv(savepath, index=False)
        return None

    @abstractmethod    
    def _compute_score(self, embeddings: np.ndarray[float], labels: np.ndarray[str]) -> Tuple[float,str]:
        """Abstract method to compute the actual score

        Args:
            embeddings (np.ndarray[float]): embeddings to calculate scores on
            labels (np.ndarray[str]): labels of the embeddings to calculate scores on; should contain only 2 unique labels

        Returns:
            float: the score 
            str: the score name
        """
        pass

    def _calculate_metrics_for_batch_and_marker(self, embeddings:np.ndarray[float], labels:np.ndarray[str],
                                                baseline_cell_line_cond:str, conditions:List[str], batch:str,
                                                marker:str)->pd.DataFrame:
        """Protected method to calculate the wanted distance metric for a given batch and marker.

        Args:
            embeddings (np.ndarray[float]): all embeddings to calculate the distance for
            labels (np.ndarray[str]): corresponding labels of the embeddings
            baseline_cell_line_cond (str): the 'cell_line_condition' that will be the baseline of the distances
            conditions (List[str]): a list of all the 'cell_line_condition' to compare to baseline
            batch (str): batch number to calculate the distance for
            marker (str): marker to calculate the distance for

        Returns:
            pd.DataFrame: updated dataframe with the results
        """
        scores = pd.DataFrame()

        baseline_reps, baseline_indices_dict = self._generate_reps_of_baseline(labels, batch, marker, baseline_cell_line_cond)
        if len(baseline_indices_dict)==0:
            logging.warning(f'No data for {batch},{marker}, {baseline_cell_line_cond}: cannot perform distance calculations!')

        baseline_labels = labels.copy()
        for rep_name in baseline_indices_dict:
            baseline_labels[baseline_indices_dict[rep_name]] = rep_name
        baseline_scores = self._calculate_metric_between_reps(embeddings, baseline_labels, baseline_reps, baseline_indices_dict)
        baseline_scores['condition'] = baseline_cell_line_cond
        scores = pd.concat([scores, baseline_scores], ignore_index=True)

        # Then we calculate the difference score between the conditions and the baseline:
        for cond in conditions:
            logging.info(f"cell line: {cond}")
            condition_reps, condition_indices_dict = self._generate_reps_of_condition_vs_baseline(labels, cond, batch, marker, baseline_cell_line_cond)
            if len(condition_indices_dict)==0:
                logging.warning(f'No data for {batch},{marker}, {cond}: cannot perform distance calculations!')
                continue
            condition_scores = self._calculate_metric_between_reps(embeddings, labels, condition_reps, condition_indices_dict)
            condition_scores['condition'] = cond
            scores = pd.concat([scores, condition_scores], ignore_index=True)
            
        scores['marker'] = marker
        scores['batch'] = batch

        return scores
    
    def _random_split_indices(self, indices:np.ndarray[int])->Tuple[np.ndarray[int],np.ndarray[int]]:
        """Randomly split indices into two parts.

        Args:
            indices (np.ndarray[int]): indices to split

        Returns:
            np.ndarray[int]: first random half of indices
            np.ndarray[int]: second random half of indices
        """
        half_size = len(indices) // 2
        part1 = np.random.choice(indices, size=half_size, replace=False)
        part2 = np.setdiff1d(indices, part1)
    
        return part1, part2
    
    def _generate_reps_of_baseline(self, labels:np.ndarray[str],
                                   batch:str, marker:str, baseline_cell_line_cond:str)->Tuple[Iterable,Dict]:
        rep1_indices = np.where(np.char.find(labels.astype(str), f'{marker}_{baseline_cell_line_cond}_{batch}_rep1')>-1)[0]
        rep2_indices = np.where(np.char.find(labels.astype(str), f'{marker}_{baseline_cell_line_cond}_{batch}_rep2')>-1)[0]

        if len(rep1_indices) == 0 and len(rep2_indices) == 0:
            logging.warn(f"Marker {marker} couldn't be found in batch {batch}. Skipping this marker..")
            return None, {}

        r1_part1, r1_part2 = self._random_split_indices(rep1_indices)
        r2_part1, r2_part2 = self._random_split_indices(rep2_indices)

        logging.info(f"Baseline split sizes: r1_part1={len(r1_part1)}, r1_part2={len(r1_part2)}, r2_part1={len(r2_part1)}, r2_part2={len(r2_part2)}")

        partial_reps = itertools.product(['rep1_part1', 'rep1_part2'], ['rep2_part1', 'rep2_part2'])
        indices_dict = {'rep1_part1': r1_part1, 'rep1_part2': r1_part2, 'rep2_part1': r2_part1, 'rep2_part2': r2_part2}

        return partial_reps, indices_dict
    
    def _generate_reps_of_condition_vs_baseline(self, labels:np.ndarray[str],
                                                cond:str, batch:str, marker:str, 
                                                baseline_cell_line_cond:str)->Tuple[Iterable,Dict]:
        reps = itertools.product(['rep1', 'rep2'], repeat=2)
        indices_dict = {}
        for repA,repB in reps:
            condition_indices = np.where(np.char.find(labels.astype(str), f'{marker}_{cond}_{batch}_{repA}')>-1)[0]
            baseline_indices = np.where(np.char.find(labels.astype(str), f'{marker}_{baseline_cell_line_cond}_{batch}_{repB}')>-1)[0]
            indices_dict[f'{repA}_cond'] = condition_indices
            indices_dict[f'{repB}_baseline'] = baseline_indices
            
            logging.info(f"{marker}_{cond}_{batch}_{repA} size: {len(condition_indices)}")
            logging.info(f"{marker}_{baseline_cell_line_cond}_{batch}_{repB} size: {len(baseline_indices)}")

            if len(condition_indices) == 0:
                logging.info(f"No samples for {marker}_{cond}_{batch}_{repA}. Skipping...")
                continue
            if len(baseline_indices) == 0:
                logging.info(f"No samples for {marker}_{baseline_cell_line_cond}_{repB}. Skipping...")
                continue
        reps = itertools.product(['rep1_cond', 'rep2_cond'], ['rep1_baseline', 'rep2_baseline'])
        return reps, indices_dict
    
    def _calculate_metric_between_reps(self, embeddings:np.ndarray[float], labels:np.ndarray[str],
                                       reps:Iterable, indices_dict:Dict)->Dict:
        """Protected method to calculate the wanted distance metric between the baseline condition and a given condition
        Args:
            embeddings (np.ndarray[float]): all embeddings to calculate the distance for
            labels (np.ndarray[str]): corresponding labels of the embeddings
            #TODO add reps and indices_dict
        Returns:
            Dict: Dictionary containing the score for each pair of reps
        """
        scores = pd.DataFrame()
        for repA,repB in reps:
            repA_indices, repB_indices = indices_dict[repA], indices_dict[repB]
            
            cur_labels = np.concatenate([labels[repA_indices],labels[repB_indices]])
            cur_embeddings = np.concatenate([embeddings[repA_indices],embeddings[repB_indices]])
            
            score, score_name = self._compute_score(cur_embeddings, cur_labels)
            score_df = pd.DataFrame(data={'repA':[repA.replace('_cond','')],'repB':[repB.replace('_baseline','')], score_name:score})
            scores = pd.concat([scores, score_df], ignore_index=True)

        return scores
    
    