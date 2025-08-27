
from abc import abstractmethod

import logging
import sys
import os
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.analysis.analyzer_distances_utils import compute_label_pair_distances_stats, get_base_to_reps
from src.analysis.analyzer_multiplex_markers import AnalyzerMultiplexMarkers
from src.analysis.analyzer import Analyzer
from src.common.utils import get_if_exists
from src.datasets.dataset_config import DatasetConfig
from src.datasets.label_utils import get_batches_from_input_folders
import numpy as np
import pandas as pd
from typing import Union, Tuple

class AnalyzerDistances(Analyzer):
    """Analyzer class is used to analyze the embeddings of a model.
    It's three main methods are:
    1. calculate(): to calculate the wanted features from the embeddings, such as UMAP or distances.
    2. load(): to load the calculated features that were previusly saved.
    3. save(): to save the calculated features.
    """
    def __init__(self, data_config: DatasetConfig, output_folder_path:str, rep_effect:bool=False,
                 multiplexed:bool=False, detailed_stats:bool=False, metric:str="euclidean", normalize_embeddings:bool=False):
        """Get an instance

        Args:
            data_config (DatasetConfig): The dataset config object. 
            output_folder_path (str): path to output folder
            rep_effect (bool, Optional): Whether to calculate distances between reps. Defaults to False.
            multiplexed (bool, Optional): Whether the embeddings are multiplexed. Defaults to False.
            detailed_stats (bool, Optional): Whether to calculate detailed stats. Defaults to False.
            metric (str, Optional): The metric to use for distance calculation. Default is "euclidean"
            normalize_embeddings (bool, Optional): Whether to normalize the embeddings before calculating distances. Defaults to False.
        """
        self.__set_params(data_config, output_folder_path)
        self.features:np.ndarray = None

        self.rep_effect = rep_effect
        self.multiplexed = multiplexed
        self.detailed_stats = detailed_stats
        self.metric = metric
        self.normalize_embeddings = normalize_embeddings

    def __set_params(self, data_config: DatasetConfig, output_folder_path:str)->None:
        """Extracting params from the configuration

        Args:
            data_config (DatasetConfig): data configuration
            output_folder_path (str): path to output folder
        """       
        self.data_config = data_config
        self.output_folder_path = output_folder_path

    def calculate(self, embeddings:np.ndarray[float], labels:np.ndarray[str])->Union[pd.DataFrame,Tuple[np.ndarray[float],np.ndarray[str]]]:
        """Calculate features from given embeddings, save in the self.features attribute and return them as well

        Args:
            embeddings (np.ndarray[float]): The embeddings
            labels (np.ndarray[str]): The corresponding labels of the embeddings
        Return:
            The calculated features
        """

        logging.info(f"[Calculate distances] data_config: {type(self.data_config)}, output_folder_path: {self.output_folder_path}, rep_effect: {self.rep_effect}, multiplexed: {self.multiplexed}, detailed_stats: {self.detailed_stats}, metric: {self.metric}")
        
        if self.multiplexed:
            logging.info("Multiplexed embeddings detected, transforming embeddings and labels using AnalyzerMultiplexMarkers.")
            analyzer_multiplex = AnalyzerMultiplexMarkers(self.data_config, self.output_folder_path)
            embeddings, labels, _ = analyzer_multiplex.calculate(embeddings, labels)  
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) 

        logging.info(f"Loaded {len(embeddings)} embeddings with {len(np.unique(labels))} unique labels.")
        logging.info(f"example label: {labels[0] if len(labels) > 0 else 'None'}")

        # Compute stats
        if not self.rep_effect:
            df_stats = compute_label_pair_distances_stats(
                embeddings=embeddings,
                labels=labels,
                metric=self.metric,
                full_stats=self.detailed_stats,
                normalize_emb=self.normalize_embeddings  
            )

            self.features = df_stats
            return self.features
        
        # Compute stats between reps
        logging.info("Calculating distances between reps.")
        base_to_reps = get_base_to_reps(labels)
        logging.info(f"Found {len(base_to_reps)} base labels with their reps.")
        logging.info(f"Example base label: {list(base_to_reps.keys())[0]} with reps {base_to_reps[list(base_to_reps.keys())[0]]}")
        all_dfs = []
        # Iterate over base labels and their reps
        for base, reps in base_to_reps.items():
            if len(reps) < 2:
                logging.warning(f"Skipping base label '{base}' with reps {reps} (less than 2 reps)")
                continue
            mask = np.isin(labels, reps)
            filtered_indices = np.where(mask)[0]
            embeddings_i = embeddings[filtered_indices]
            labels_i = labels[filtered_indices]

            df_part = compute_label_pair_distances_stats(
                embeddings=embeddings_i,
                labels=labels_i,
                metric=self.metric,
                full_stats=self.detailed_stats,
                normalize_emb=self.normalize_embeddings 
            )
            all_dfs.append(df_part)

        df_stats = pd.concat(all_dfs, ignore_index=True)

        df_stats = self.__add_rep_and_label_to_df(df_stats)
        df_stats['rep_effect'] = df_stats.apply(lambda row: self.__detect_suspected_rep_effect(row, df_stats), axis=1)

        self.features = df_stats
        return self.features 

    def save(self)->None:
        """
        Save the calculated distances to a specified file.
        """
        savepath = self.__get_save_path()
        
        logging.info(f"Saving distances to {savepath}")

        self.features.to_csv(savepath, index=False)
        
        logging.info(f"Saved distance stats to {savepath}")

    def load(self)->None:
        """
        Load the calculated distances from a specified file.
        """
        loadpath = self.__get_save_path()
        
        logging.info(f"Loading distances from {loadpath}")

        self.features = pd.read_csv(loadpath)

        logging.info(f"Loaded distance stats from {loadpath}")

    def __get_save_path(self)->str:
        """Get the path to save the features

        Returns:
            str: The path to save the features
        """
        
        output_folder_path = self.get_saving_folder(feature_type='distances')
        os.makedirs(output_folder_path, exist_ok=True)
        filename = f"distances_stats_{self.metric}{'_detailed' if self.detailed_stats else ''}{'_rep' if self.rep_effect else ''}{'_multiplexed' if self.multiplexed else ''}.csv"
        
        path = os.path.join(output_folder_path, filename)

        return path
    
    def __add_rep_and_label_to_df(self, df:pd.DataFrame)->pd.DataFrame:
        # extract the numeric replicate from each label column
        df['rep1'] = df['label1'].str.extract(r'_rep(\d+)$').astype(int)
        df['rep2'] = df['label2'].str.extract(r'_rep(\d+)$').astype(int)
        # strip the “_repN” suffix to get the common label
        df['label'] = df['label1'].str.replace(r'_rep\d+$', '', regex=True)
        cols = ['rep1', 'rep2', 'label'] + [c for c in df.columns if c not in ('rep1','rep2','label')]
        return df[cols]
    
    def __detect_suspected_rep_effect(self, row, df, dist='p50', delta_frac=0.1):
        """
        Detect replicate effects by comparing inter-replicate distances to intra-replicate baselines.
        
        Parameters:
        -----------
        row : pd.Series
            Row containing 'label', 'rep1', 'rep2' and distance percentiles
        df : pd.DataFrame
            Full dataframe with all replicate comparisons
        dist : str
            Distance column to use (default 'p50')
        delta_frac : float
            Fraction of IQR to add as threshold (default 0.1)
        
        Returns:
        --------
        bool, str, or None
            True/False if rep effect detected/not detected
            'missing_rep' if one replicate is missing
            'imbalanced' if inter group has 100x more data than the intra groups
            None for same-replicate comparisons
        """
        
        # Skip same-replicate comparisons
        if row.rep1 == row.rep2:
            return None
        
        # Get intra-replicate distances for rep1 (rep1 vs rep1)
        intra1 = df[
            (df.label == row.label) &
            (df.rep1 == row.rep1) &
            (df.rep2 == row.rep1)
        ]
        
        # Get intra-replicate distances for rep2 (rep2 vs rep2)  
        intra2 = df[
            (df.label == row.label) &
            (df.rep1 == row.rep2) &
            (df.rep2 == row.rep2)
        ]
        
        # Check if either replicate is missing
        if intra1.empty or intra2.empty:
            return 'missing_rep'
        
        # Validate exactly one row per replicate before taking .iloc[0]
        if len(intra1) != 1 or len(intra2) != 1:
            return 'validation_error'
        
        intra1 = intra1.iloc[0]
        intra2 = intra2.iloc[0]
        
        # Get the inter-replicate distance value
        inter_val = row[dist]
        
        # Check for imbalanced groups (100x difference)
        if intra1['total_pairs'] >= 100 * row['total_pairs'] or intra2['total_pairs'] >= 100 * row['total_pairs']:
            return 'imbalanced'
        
        # Adaptive deltas based on IQR
        delta1 = delta_frac * (intra1['p75'] - intra1['p25'])
        delta2 = delta_frac * (intra2['p75'] - intra2['p25'])
        intra1_thresh = intra1[dist] + delta1
        intra2_thresh = intra2[dist] + delta2
        
        return inter_val > max(intra1_thresh, intra2_thresh)
