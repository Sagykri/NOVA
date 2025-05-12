import sys
import os
from typing import Dict, Tuple

sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.common.utils import get_if_exists
from src.datasets.label_utils import map_labels, split_markers_from_labels
from src.datasets.dataset_config import DatasetConfig

from src.analysis.analyzer_umap import AnalyzerUMAP
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

class AnalyzerUMAPMultiplexMarkers(AnalyzerUMAP):

    __COLUMN_MARKER = 'Marker' # Of type string
    __COLUMN_PHENOTYPE = 'Phenotype' # Of type string
    __COLUMN_EMBEDDINGS = 'Embeddings' # Of type np.ndarray[float]
    __COLUMN_PATHS = 'Path'

    def __init__(self, data_config: DatasetConfig, output_folder_path:str):
        super().__init__(data_config, output_folder_path)


    def calculate(self, embeddings:np.ndarray[float], labels:np.ndarray[str], paths:np.ndarray[str]=None)->Tuple[np.ndarray[float],np.ndarray[str], Dict[str,float]]:
        """Calculate UMAP embeddings for multiplexed embeddings from the given embeddings and store the results in the `self.features` attribute. 
         This method takes in embeddings and their associated labels, and computes multiplexed embeddings by grouping the data based on shared phenotypes.

        Args:
            embeddings (np.ndarray[float]): The input embeddings with shape (n_samples, n_features).
            labels (np.ndarray[str]): The labels associated with each embedding.
            paths (np.ndarray[str]): The image paths associated with each embedding.
        Returns:
            Tuple[np.ndarray[float], np.ndarray[str]]: 
                - The UMAP embeddings after dimensionality reduction with shape (n_mutliplexed_samples, n_components).
                - The corresponding phenotypes labels preserving the association with the UMAP embeddings.
                - A dictionary with 'ari' as key and the ari score as value
                - The corresponding paths preserving the association with the UMAP embeddings.
        """
        
        logging.info(f"[AnalyzerUMAPMultiplexMarkers.calculate] Embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}, Paths shape: {paths.shape}")
        df = self.__format_embeddings_to_df(embeddings, labels, paths)
        multiplexed_embeddings, multiplexed_labels, multiplexed_paths = self.__get_multiplexed_embeddings(df)
        logging.info(f"[AnalyzerUMAPMultiplexMarkers.calculate] Multiplexed embeddings shape: {multiplexed_embeddings.shape}, Labels shape: {multiplexed_labels.shape}, Paths shape: {multiplexed_paths.shape}")

        logging.info(f"[AnalyzerUMAPMultiplexMarkers.calculate] Calculating UMAP of multiplex embeddings")
        umap_embeddings = self._compute_umap_embeddings(multiplexed_embeddings)     
        
        if self.data_config.SHOW_ARI:
            multiplexed_labels_for_ari = map_labels(multiplexed_labels, self.data_config, self.data_config, config_function_name='ARI_LABELS_FUNC')
            ari = self._compute_ari(multiplexed_embeddings, multiplexed_labels_for_ari)
            ari_score = {'ari':ari}

        else:
            ari_score = {}

        self.features = umap_embeddings
        self.labels = multiplexed_labels
        self.ari_scores = ari_score
        self.paths = multiplexed_paths

        return umap_embeddings, multiplexed_labels, multiplexed_paths, ari_score
    

    def __format_embeddings_to_df(self, embeddings:np.ndarray[float], labels: np.ndarray[str], paths: np.ndarray[str] = None)->pd.DataFrame:
        """Format the embeddings into a Dataframe holding three columns:\n
        self.__COLUMN_MARKER:str\n
        self.__COLUMN_PHENO:str\n
        self.__COLUMN_EMBEDDINGS:np.ndarray[float]
        self.__COLUMN_PATHS:str

        Args:
            embeddings (np.ndarray[float]): The embeddings to format
            labels (np.ndarray[str]): The labels for each embedding holding the marker name and phenotype
            paths (np.ndarray[str]): The paths for tiles' locations

        Returns:
            pd.DataFrame: The formatted embeddings as a dataframe
        """
        
        # Dataframe for holding the marker name and phenotype
        labels_df = pd.DataFrame([self.__split_label_to_marker_and_pheno(l) for l in labels])
        # Holds the embeddings
        embeddings_series = pd.DataFrame({self.__COLUMN_EMBEDDINGS: [*embeddings]})
        # Merge them together to a unite dataframe
        df = pd.merge(labels_df, embeddings_series, left_index=True, right_index=True)

        # Optionally add paths
        if paths is not None:
            if len(paths) != len(df):
                raise ValueError("Length of paths does not match number of embeddings")
            df[self.__COLUMN_PATHS] = list(paths)
        
        return df
    
    def __split_label_to_marker_and_pheno(self, label:str)->Dict[str, str]:
        """Split label to marker and phenotype

        Args:
            label (str): The label to split

        Returns:
            Dict[str, str]: The marker and phenotype in these keys: self.__COLUMN_MARKER, self.__COLUMN_PHENO
        """
        
        marker, phenotype = split_markers_from_labels([label], self.data_config)
        
        # Since it returned lists of single element, we take the first one
        marker = marker[0]
        phenotype = phenotype[0]
            
        return {
                    self.__COLUMN_MARKER: marker,
                    self.__COLUMN_PHENOTYPE: phenotype
                }
    
    def __get_multiplexed_embeddings(self, embeddings_df:pd.DataFrame)->Tuple[np.ndarray[float], np.ndarray[str]]:
        """Get multiplexed embeddings and their corresponding labels

        Args:
            embeddings_df (pd.DataFrame): The dataframe holding the marker, phenotype and embeddings to multiplexed

        Returns:
            Tuple[np.ndarray[float], np.ndarray[str]]: The multiplexed embeddings and their labels
        """
        
        # Get common markers between all groups (since we want to use only them)
        common_markers = self.__get_common_markers_between_groups(embeddings_df)
        logging.info(f"[AnalyzerUMAPMultiplexMarkers.calculate] Common markers: {common_markers}")
        
        # calculate the multiplexed embeddings:
        # Take only common markers, group by phenotype and concatenate the embeddings within groups
        result_df = embeddings_df\
                        [embeddings_df[self.__COLUMN_MARKER].isin(common_markers)]\
                        .groupby(self.__COLUMN_PHENOTYPE)\
                        .apply(self.__get_multiplexed_embeddings_for_phenotype_group)\
                        .reset_index(drop=True)
        
        all_multiplexed_embeddings  = np.vstack(result_df[self.__COLUMN_EMBEDDINGS].to_numpy())
        all_labels                  = np.concatenate(result_df[self.__COLUMN_PHENOTYPE].to_numpy())\
                                                                                .astype(object)
        
        if self.__COLUMN_PATHS in result_df:
            all_paths = np.concatenate(result_df[self.__COLUMN_PATHS].to_numpy()).astype(object)
        else:
            all_paths = np.array([None] * len(all_labels))

        
        return all_multiplexed_embeddings, all_labels, all_paths
    
    def __get_common_markers_between_groups(self, df:pd.DataFrame)->np.ndarray[str]:
        """Get the common markers between the groups in the given dataframe

        Args:
            df (pd.DataFrame): The dataframe holding the marker, phenotype and embeddings

        Returns:
            np.ndarray[str]: The common markers
        """
        df_grouped = df.groupby(self.__COLUMN_PHENOTYPE)
        unique_markers_set = map(set, df_grouped[self.__COLUMN_MARKER].unique())
        common_markers = set.intersection(*unique_markers_set)
        common_markers = np.asarray(list(common_markers))
        
        return common_markers
    
    def __get_multiplexed_embeddings_for_phenotype_group(self, phenotype_group: pd.api.typing.DataFrameGroupBy)->pd.Series:
        """Concatenate the embeddings of all markers for all subgroups in the given phenotype group

        Args:
            phenotype_group (pd.api.typing.DataFrameGroupBy): The group holds all samples for a single phenotype

        Returns:
            pd.Series: Two columns: self.__COLUMN_PHENOTYPE for the phenotypes, self.__COLUMN_EMBEDDINGS for the multiplexed embeddings
        """
        
        # Get the phenotype of this group
        phenotypes = phenotype_group[self.__COLUMN_PHENOTYPE]
        
        assert len(np.unique(phenotypes)) == 1, "Expected to have only a single unique phenotype"
        # Get the phenotype (since all are the same, take the first one)
        phenotype = phenotypes.iloc[0]
        
        logging.info(f"[AnalyzerUMAPMultiplexMarkers.calculate] Phenotype: {phenotype}")
        
        # Determine the number of subgroups to be created (based on the smallest marker group)
        n_subgroups = phenotype_group[self.__COLUMN_MARKER].value_counts().min()
        logging.info(f"[AnalyzerUMAPMultiplexMarkers.calculate] Detected {n_subgroups} subgroups")
        
        # List of multiplexed embeddings per subgroups within the given phenotype group 
        multiplexed_embeddings, multiplexed_paths = zip(*[
            self.__get_multiplexed_embeddings_for_next_phenotype_subgroup(phenotype_group)
            for _ in tqdm(range(n_subgroups))
        ])

        multiplexed_embeddings = np.array(multiplexed_embeddings, dtype=object)
        multiplexed_paths = np.array(multiplexed_paths, dtype=object)
                
        # Repeat the phenotype to match the len of multiplexed_embeddings
        phenotype_repeated = np.full(len(multiplexed_embeddings), phenotype)
        
        assert len(multiplexed_embeddings) == len(phenotype_repeated), f"Multiplexed embeddings and phentoypes numbers must match, but len(multiplexed_embeddings)={len(multiplexed_embeddings)}, len(phenotype_repeated)={len(phenotype_repeated)}"
        
        logging.info(f"[AnalyzerUMAPMultiplexMarkers.calculate] [{phenotype}] multiplexed embeddings shape: {multiplexed_embeddings.shape} phenotype_repeated shape: {phenotype_repeated.shape}")
        
        return pd.Series({
            self.__COLUMN_PHENOTYPE: phenotype_repeated, 
            self.__COLUMN_EMBEDDINGS: multiplexed_embeddings,
            self.__COLUMN_PATHS: multiplexed_paths
        })
        
    def __get_multiplexed_embeddings_for_next_phenotype_subgroup(self, phenotype_group:pd.api.typing.DataFrameGroupBy)->np.ndarray[float]:
        """Concatenate the embeddings of a single subgroups within the phenotype_group

        Args:
            phenotype_group (pd.api.typing.DataFrameGroupBy):  The group holds all samples for a single phenotype

        Returns:
            np.ndarray[float]: The multiplexed embeddings for a single subgroup
        """
        # Take a single sample per marker (sorted by the same markers order)
        subgroup = phenotype_group\
                    .groupby(self.__COLUMN_MARKER)\
                    .sample(n=1, replace=False, random_state=self.data_config.SEED)\
                    .sort_values(self.__COLUMN_MARKER)
                
        # Concatenate the embeddings of all markers in this subgroup
        multiplexed_embeddings = np.concatenate(subgroup[self.__COLUMN_EMBEDDINGS].to_numpy())
        multiplexed_paths = subgroup[self.__COLUMN_PATHS].tolist() if self.__COLUMN_PATHS in subgroup.columns else [None] * len(subgroup)

        
        # Remove this subgroup from the pool of subgroups to analyze
        phenotype_group.drop(index=subgroup.index, inplace=True)
        
        return multiplexed_embeddings, multiplexed_paths