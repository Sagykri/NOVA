
from collections import defaultdict
import logging
import os
from pathlib import Path
import re
from typing import Dict, List, Tuple
import numpy as np
import sys

import torch

sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.preprocessing.path_utils import get_filename
from src.datasets.dataset_NOVA import DatasetNOVA
from src.datasets.dataset_config import DatasetConfig

class DatasetNOVAMC(DatasetNOVA):
    """
    Dataset customize to load the data ordered in NOVA's files structure
    """
    
    def __init__(self, dataset_config: DatasetConfig):    
        # Set random seed
        np.random.seed(dataset_config.SEED)
        self.dataset_config:DatasetConfig = dataset_config
        X_paths, y = self._load_grouped_data_paths()
        self.set_Xy(X_paths, y)
        self.set_transform(None)
        

    def _load_grouped_data_paths(self)-> Tuple[np.ndarray[str], np.ndarray[str]]:

        image_paths, labels = self._load_data_paths()
        groups = defaultdict(dict)

        for image_path in image_paths:
            if type(image_path) is not Path:
                image_path = Path(image_path)
            # Take the path of the file's grandfather, i.e. the root folder of markers
            markers_rootfolder_path = str(image_path.parent.parent)
            # Clean it a bit - remove the input folders paths
            pattern = '|'.join(map(re.escape, [os.path.dirname(path) for path in self.dataset_config.INPUT_FOLDERS]))
            markers_rootfolder_path = re.sub(pattern, '', markers_rootfolder_path)

            # Extract identifier from the file name
            image_id = get_filename(image_path).split('_')[3] 
            rep_id = get_filename(image_path).split('_')[0]
            panel_id = get_filename(image_path).split('_')[4]
            # Get group id
            group_id = '_'.join([markers_rootfolder_path.replace(os.sep,'_'), rep_id, panel_id, image_id])
            
            # Get marker name to serve as key
            marker_name = image_path.parent.name
            
            # Group by id and store the file path under the extracted marker name
            groups[group_id].update({marker_name: str(image_path)})
            
        labels = list(groups.keys())
        grouped_paths = np.asarray([list(groups[label].values()) for label in labels])
        labels = np.asarray(['_'.join(label.split('_')[1:-2]) for label in labels])
        
        return grouped_paths, labels



    def __getitem__(self, index:int)->Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
            """Get item given an batch index

            Args:
                index (int): The batch index

            Returns:
                Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
                    - The data
                    - The labels
                    - The path to the data
            """
            current_paths, label = self.X_paths[index], self.y[index]

            Xs = []
            for i, path in enumerate(current_paths):
                 
                # Load X from path
                X = np.load(path)
                # Move the channels axis from being the last to be second
                X = np.moveaxis(X, -1, 1)
                
                # Check the shape of X is valid
                assert X.shape[1] == self.dataset_config.NUM_CHANNELS, f"Number of channels expected to be {self.dataset_config.NUM_CHANNELS} but got {X.shape[1]}"
                assert X.shape[2:] == self.dataset_config.IMAGE_SIZE, f"Image size expected to be {self.dataset_config.IMAGE_SIZE} but got {X.shape[2:]}"
                if i!=len(current_paths)-1:
                     X = X[:,:1,:,:]
                Xs.append(X)

            X = np.concatenate(Xs, axis=1)
            assert X.shape[1] == len(current_paths)+1
            # Load y with the label id
            y = np.full(len(X), self.label2id(label))
            
            # ToTensor
            X,y = torch.from_numpy(X).float(), torch.from_numpy(y).int()
            
            # Apply transform if exists
            if self.transform is not None:
                # Apply transform for each sample in X
                X = torch.stack(
                        [self.transform(x) for x in torch.unbind(X, dim=0)]
                    , dim=0)
                
            
            path = np.full(len(X), label)
            return X, y, path