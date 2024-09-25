
import sys
import os

sys.path.insert(1, os.getenv("NOVA_HOME"))

from abc import abstractmethod
import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Iterable, Dict
import itertools

from src.datasets.label_utils import get_unique_parts_from_labels, get_cell_lines_conditions_from_labels, get_markers_from_labels, get_batches_from_labels
from src.datasets.dataset_config import DatasetConfig
from src.analysis.analyzer import Analyzer
from src.common.utils import get_if_exists

class AnalyzerDistances(Analyzer):
    """
    AnalyzerDistances is responsible for calculating distance metrics between different conditions
    based on model embeddings. The distances are computed for each marker and batch, comparing
    a baseline condition to other conditions.
    """
    def __init__(self, data_config: DatasetConfig, output_folder_path:str):
        """Get an instance

        Args:
            data_config (DatasetConfig): The dataset configuration object. 
            output_folder_path (str): path to output folder
        """
        super().__init__(data_config, output_folder_path)


    def calculate(self, embeddings:np.ndarray[float], labels:np.ndarray[str])->pd.DataFrame:
        """Calculate distance metrics from given embeddings and labels.

        Args:
            embeddings (np.ndarray[float]): The embeddings
            labels (np.ndarray[str]): The corresponding labels of the embeddings
        Returns:
            pd.DataFrame: DataFrame containing calculated distances with columns:
            - 'marker': The marker for which the distance was calculated on.
            - 'batch': The batch for which the distance was calculated on.
            - 'repA': The rep of the condition data
            - 'repB': The rep of the baseline data
            - 'distance_metric': The calculated distance metric (specific name is defined by the specific distance metric).
        """
        # First we must define what is the baseline cell line condition of the calculation: 
        # we always calculate the distance for a given marker between one condition and the baseline condition.
        baseline_cell_line_cond = get_if_exists(self.data_config, 'BASELINE_CELL_LINE_CONDITION', None)
        assert baseline_cell_line_cond is not None, "BASELINE_CELL_LINE_CONDITION is None. You have to specify the baseline to calculate the distance score against (for example: WT_Untreated or TDP43_Untreated)"

        # Next we extract info from our labels: which other conditions we have, which unique markers and batches we can work with
        conditions = self._get_conditions_for_distances(labels)
        markers = get_unique_parts_from_labels(labels, get_markers_from_labels)
        batches = get_unique_parts_from_labels(labels, get_batches_from_labels, self.data_config)
        logging.info(f"[AnalyzerDistances.calculate] Conditions: {conditions}, markers: {markers}, batches: {batches}, baseline_cell_line_cond: {baseline_cell_line_cond}")

        # Finally we can calculate the distances, separatly for each batch and marker
        scores = []
        for batch in batches:
            logging.info(f"[AnalyzerDistances.calculate] batch: {batch}")
            for marker in markers:   
                logging.info(f"[AnalyzerDistances.calculate] marker: {marker}")
                new_scores = self._calculate_metrics_for_baseline(embeddings, labels, baseline_cell_line_cond,
                                                                  batch, marker)
                scores.append(new_scores)
                for cond in conditions:
                    new_scores = self._calculate_metrics_for_condition_vs_baseline(embeddings, labels, 
                                                                                   baseline_cell_line_cond,
                                                                                   cond, batch, marker)
                    scores.append(new_scores)

        scores = pd.concat(scores, ignore_index=True)

        self.features = scores

        return scores
    
    def load(self)->None:
        """
        Load pre-calculated distances from a file into the self.features attribute.
        """
        output_folder_path = self.get_saving_folder(feature_type='distances')
        logging.info(f"[save scores]: output_folder_path: {output_folder_path}")
        loadpath = self._get_save_path(output_folder_path)
        self.features = pd.read_csv(loadpath)
        return None

    def save(self)->None:
        """"
        Save the calculated distances to a specified file.
        """
        output_folder_path = self.get_saving_folder(feature_type='distances')
        savepath = self._get_save_path(output_folder_path)
        logging.info(f"Saving scores to {savepath}")
        self.features.to_csv(savepath, index=False)
        return None

    @abstractmethod    
    def _compute_score(self, embeddings: np.ndarray[float], labels: np.ndarray[str]) -> Tuple[float,str]:
        """
        Abstract method to compute the score between two sets of embeddings.

        Args:
            embeddings (np.ndarray[float]): The embeddings to compute scores on.
            labels (np.ndarray[str]): Corresponding labels; should contain only 2 unique labels.

        Returns:
            float: The calculated score.
            str: Name of the score metric.
        """
        pass

    def _calculate_metrics_for_baseline(self, embeddings:np.ndarray[float], labels:np.ndarray[str],
                                                baseline_cell_line_cond:str, batch:str,
                                                marker:str)->pd.DataFrame:
        """
        Calculate metrics for the baseline samples.

        Args:
            embeddings (np.ndarray[float]): All embeddings to calculate the distance for.
            labels (np.ndarray[str]): Corresponding labels for the embeddings.
            baseline_cell_line_cond (str): The baseline condition to compare against, example: 'WT_Untreated;
            batch (str): batch identifier to calculate the distance for, example: 'batch6'
            marker (str): marker to calculate the distance for, example: 'G3BP1'

        Returns:
            pd.DataFrame: DataFrame with calculated scores.
        """
        # We want to calculate the baseline distances: the distances between the baseline to itself.
        # These distances will be compared to the baseline vs condition distances, and they represent the inherent variance in the data (including rep effect).
        # First step is to generate the artifical splits of the two baseline reps:
        baseline_reps, baseline_indices_dict = self._generate_reps_of_baseline(labels, batch, marker, baseline_cell_line_cond)
        if len(baseline_indices_dict)==0:
            logging.warning(f'No data for {batch},{marker}, {baseline_cell_line_cond}: cannot perform distance calculations!')
            return None
        
        # We want the baseline samples to have an artificial labels, dervied from the splits
        baseline_labels = labels.copy()
        for rep_name in baseline_indices_dict:
            baseline_labels[baseline_indices_dict[rep_name]] = rep_name
        # Then we can calculate the distances between the splits
        baseline_scores = self._calculate_metric_between_reps(embeddings, baseline_labels, baseline_reps, baseline_indices_dict)
        baseline_scores['condition'] = baseline_cell_line_cond
        baseline_scores['marker'] = marker
        baseline_scores['batch'] = batch
        
        return baseline_scores
    

    def _calculate_metrics_for_condition_vs_baseline(self, embeddings:np.ndarray[float], labels:np.ndarray[str],
                                                baseline_cell_line_cond:str, cond:str, batch:str,
                                                marker:str)->pd.DataFrame:
        """
        Calculate metrics between baseline samples and the condition samples.

        Args:
            embeddings (np.ndarray[float]): All embeddings to calculate the distance for.
            labels (np.ndarray[str]): Corresponding labels for the embeddings.
            baseline_cell_line_cond (str): The baseline condition to compare against, example: 'WT_Untreated;
            cond (str): The condition to compate against the baseline., example: 'WT_stress'
            batch (str): batch identifier to calculate the distance for, example: 'batch6'
            marker (str): marker to calculate the distance for, example: 'G3BP1'

        Returns:
            pd.DataFrame: DataFrame with calculated scores.
        """
        logging.info(f"cell line: {cond}")
        condition_reps, condition_indices_dict = self._generate_reps_of_condition_vs_baseline(labels, cond, batch, marker, baseline_cell_line_cond)
        if len(condition_indices_dict)==0:
            logging.warning(f'No data for {batch},{marker}, {cond}: cannot perform distance calculations!')
            return None
        condition_scores = self._calculate_metric_between_reps(embeddings, labels, condition_reps, condition_indices_dict)
        condition_scores['condition'] = cond
        condition_scores['marker'] = marker
        condition_scores['batch'] = batch
        
        return condition_scores
    
    def _random_split_indices(self, indices:np.ndarray[int])->Tuple[np.ndarray[int],np.ndarray[int]]:
        """
        Randomly split indices into two parts.

        Args:
            indices (np.ndarray[int]): Indices to split

        Returns:
            Tuple[np.ndarray[int], np.ndarray[int]]: Two random splits of the indices.

        """
        np.random.seed(self.data_config.SEED)
        half_size = len(indices) // 2
        part1 = np.random.choice(indices, size=half_size, replace=False)
        part2 = np.setdiff1d(indices, part1)
    
        return part1, part2
    
    def _get_conditions_for_distances(self, labels:np.ndarray[str])->np.ndarray[str]:
        """
        Get conditions to calculate distances for, excluding the baseline condition.

        Args:
            labels (np.ndarray[str]): Labels containing the conditions.

        Returns:
            np.ndarray[str]: Conditions excluding the baseline condition.
        """
        conditions = get_unique_parts_from_labels(labels, get_cell_lines_conditions_from_labels, self.data_config)
        conditions = np.delete(conditions, np.where(conditions == self.data_config.BASELINE_CELL_LINE_CONDITION)[0]) # Remove the baseline
        return conditions

    def _generate_reps_of_baseline(self, labels:np.ndarray[str],
                                   batch:str, marker:str, baseline_cell_line_cond:str)->Tuple[Iterable,Dict]:
        """
        Generate random splits for baseline replicates.

        This function identifies the indices for two replicates (`rep1` and `rep2`) of a given marker under the baseline
        condition, for the specified batch. Each replicate is further randomly split into two parts. The function then
        returns all possible pairwise combinations between parts of `rep1` and parts of `rep2`.

        Args:
            labels (np.ndarray[str]): Labels of the samples.
            batch (str): The batch identifier to filter the samples.
            marker (str): The marker identifier to filter the samples.
            baseline_cell_line_cond (str): The baseline condition identifier to filter the samples.

        Returns:
            Tuple[Iterable, Dict]: 
                - An iterable of pairwise combinations between `rep1` and `rep2` parts (i.e., `rep1_part1` with `rep2_part1`, `rep1_part1` with `rep2_part2`, etc.).
                - A dictionary where the keys are `rep1_part1`, `rep1_part2`, `rep2_part1`, `rep2_part2` and the values 
                are the corresponding indices in the `labels` array.

        Raises:
            Warning: If no replicates for the given marker and batch are found, a warning is logged, and the function returns 
            `None` and an empty dictionary.
        """
        rep1_indices = np.where(np.char.find(labels.astype(str), f'{marker}_{baseline_cell_line_cond}_{batch}_rep1')>-1)[0]
        rep2_indices = np.where(np.char.find(labels.astype(str), f'{marker}_{baseline_cell_line_cond}_{batch}_rep2')>-1)[0]

        if len(rep1_indices) == 0 or len(rep2_indices) == 0:
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
                                                baseline_cell_line_cond:str,
                                                first_rep:str = 'rep1',
                                                second_rep:str = 'rep2')->Tuple[Iterable,Dict]:
        """
        Generate combinations of replicates for condition vs baseline.

        Args:
            labels (np.ndarray[str]): Labels of the dataset.
            cond (str): Condition to compare.
            batch (str): Batch identifier.
            marker (str): Marker identifier.
            baseline_cell_line_cond (str): Baseline condition.

        Returns:
            Tuple[Iterable, Dict]: 
                - An iterable of pairwise combinations between baseline rep and condition rep (i.e., `rep1_baseline` with `rep2_condition`, etc.).
                - A dictionary where the keys are `rep1_cond`, `rep1_baseline`, `rep2_cond`, `rep2_baseline` and the values 
                    are the corresponding indices in the `labels` array.
        """
        reps_combinations = itertools.product([first_rep, second_rep], repeat=2)
        indices_dict = {}
        reps = []
        for repA,repB in reps_combinations:
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
            
            indices_dict[f'{repA}_cond'] = condition_indices
            indices_dict[f'{repB}_baseline'] = baseline_indices
            reps.append([f'{repA}_cond',f'{repB}_baseline'])

        return reps, indices_dict
    
    def _calculate_metric_between_reps(self, embeddings:np.ndarray[float], labels:np.ndarray[str],
                                       reps:Iterable, indices_dict:Dict)->pd.DataFrame:
        """
        Calculate the metric between replicates.

        Args:
            embeddings (np.ndarray[float]): All embeddings to calculate the distance for.
            labels (np.ndarray[str]): Corresponding labels for the embeddings.
            reps (Iterable): Replicates to compare.
            indices_dict (Dict[str, np.ndarray[int]]): Indices for the replicates.
        Returns:
            pd.DataFrame: DataFrame with calculated scores.
        """
        scores = pd.DataFrame()
        for repA,repB in reps:
            repA_indices, repB_indices = indices_dict[repA], indices_dict[repB]
            
            cur_labels = np.concatenate([labels[repA_indices],labels[repB_indices]])
            cur_embeddings = np.concatenate([embeddings[repA_indices],embeddings[repB_indices]])
            
            score, score_name = self._compute_score(cur_embeddings, cur_labels)
            score_df = pd.DataFrame(data={'repA':[repA],'repB':[repB], score_name:score})
            scores = pd.concat([scores, score_df], ignore_index=True)

        return scores
    
    