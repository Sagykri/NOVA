from abc import abstractmethod

from src.common.configs.dataset_config import DatasetConfig
from src.common.configs.trainer_config import TrainerConfig ## TODO SAGY CHANGE

from src.Analysis.analyzer import Analyzer
from src.common.lib.utils import handle_log

import logging
import os
import numpy as np
import umap
class AnalyzerUMAP(Analyzer):
    def __init__(self, trainer_conf: TrainerConfig, data_conf: DatasetConfig):
        super().__init__(trainer_conf, data_conf)


    @abstractmethod
    def calculate(self, embeddings:np.ndarray, labels:np.ndarray)->None:
        """Calculate UMAP embeddings from given embeddings, save in the self.features attribute

        Args:
            embeddings (np.ndarray): The embeddings
            labels (np.ndarray): The corresponding labels of the embeddings
        """
        pass
    
    
    def load(self, umap_type:str)->None:
        """load the saved UMAP embeddings into the self.features attribute

        Args:
            umap_type (str): string indicating the umap type ('umap0','umap1','umap2')
        """
        model_output_folder = self.output_folder_path

        output_folder_path = os.path.join(model_output_folder, 'figures', self.experiment_type,'UMAP', umap_type)
        if not os.path.exists(output_folder_path):
            logging.info(f"{output_folder_path} doesn't exists. Can't load!")
            return None

        title = f"{'_'.join([os.path.basename(f) for f in self.input_folders])}_{'_'.join(self.reps)}"
        saveroot = os.path.join(output_folder_path,f'{title}')
        if not os.path.exists(saveroot):
            logging.info(f"{saveroot} doesn't exists. Can't load!")
            return None
        
        self.features = np.load(f'{saveroot}_{umap_type}.npy')
    
    def save(self, umap_type:str)->None:
        """save the calculated UMAP embeddings in path derived from self.output_folder_path

        Args:
            umap_type (str): string indicating the umap type ('umap0','umap1','umap2')
        """
        model_output_folder = self.output_folder_path

        output_folder_path = os.path.join(model_output_folder, 'figures', self.experiment_type,'UMAP', umap_type)
        if not os.path.exists(output_folder_path):
            logging.info(f"{output_folder_path} doesn't exists. Creating it")
            os.makedirs(output_folder_path, exist_ok=True)

        title = f"{'_'.join([os.path.basename(f) for f in self.input_folders])}_{'_'.join(self.reps)}"
        saveroot = os.path.join(output_folder_path,f'{title}')
        if not os.path.exists(saveroot):
            os.makedirs(saveroot, exist_ok=True)
        
        np.save(f'{saveroot}_{umap_type}.npy', self.features)

    def _compute_umap_embeddings(self, embeddings:np.ndarray, 
                                 n_neighbors:int=15, min_dist:float=0.1,
                                 n_components:int=2, random_state:int=42)->np.ndarray:
        """Protected method to calculate UMAP reduction given embeddings

        Args:
            embeddings (np.ndarray): embeddings to calculate UMAP on
            
            n_neighbors (int, optional, default 15): The size of local neighborhood
            (in terms of number of neighboring sample points) used for manifold approximation.
            Larger values result in more global views of the manifold, while smaller values result
            in more local data being preserved. In general values should be in the range 2 to 100.
            
            min_dist (float, optional, deafult 0.1): The effective minimum distance between embedded points.
            Smaller values will result in a more clustered/clumped embedding where nearby points on the manifold
            are drawn closer together, while larger values will result on a more even dispersal of points.
            The value should be set relative to the spread value, which determines the scale at which
            embedded points will be spread out.
            
            n_components (int, optional, default 2):The dimension of the space to embed into. This defaults to 2 
            to provide easy visualization, but can reasonably be set to any integer value in the range 2 to 100.
            
            random_state (int, optional, default 42): If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by np.random.

        Returns:
            np:ndarray: the UMAP embeddings
        """
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, 
                            n_components=n_components, random_state=random_state)
        umap_embeddings = reducer.fit_transform(embeddings)
        return umap_embeddings