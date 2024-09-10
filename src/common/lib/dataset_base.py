
from abc import abstractmethod

import logging
import sys
import os
from typing import Iterable, List, Tuple, Union
import numpy as np
import torch
from copy import deepcopy

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from sklearn.model_selection import train_test_split
from src.common.configs.dataset_config import DatasetConfig
import src.common.lib.utils as utils
from copy import deepcopy
from torchvision import transforms

class DatasetBase(torch.utils.data.Dataset):    
    def __init__(self, dataset_config: DatasetConfig):
        """Init a new dataset object while loading the paths and labels

        Args:
            dataset_config (DatasetConfig): The dataset configuration
        """
        self.dataset_config:DatasetConfig = dataset_config
        X_paths, y = self._load_data_paths()
        # X_paths, y, _ = self._load_data_paths()
        self.set_Xy(X_paths, y)
        self.set_transform(None)
        
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
        current_path, label = self.X_paths[index], self.y[index]
        
        # Load X from path
        X = np.load(current_path)
        # Move the channels axis from being the last to be second
        X = np.moveaxis(X, -1, 1)
        
        # Check the shape of X is valid
        assert X.shape[1] == self.dataset_config.NUM_CHANNELS, f"Number of channels expected to be {self.dataset_config.NUM_CHANNELS} but got {X.shape[1]}"
        assert X.shape[2:] == self.dataset_config.IMAGE_SIZE, f"Image size expected to be {self.dataset_config.IMAGE_SIZE} but got {X.shape[2:]}"
        
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
            
        
        path = np.full(len(X), current_path)
            
        return X, y, path     
        
    def __len__(self)->int:
        """Return the len of the dataset

        Returns:
            int: The len
        """
        return len(self.y)
        
    @staticmethod
    def get_collate_fn(shuffle:bool=False):
        """Get the collate function for the dataloader

        Args:
            shuffle (bool, optional): Should shuffle the data?. Defaults to False.
        """
        def collate_fn(batch:List[Tuple[torch.Tensor, torch.Tensor, np.ndarray]])->Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
            """Stack the X, y and paths for the entire batch

            Args:
                batch (List[Tuple[torch.Tensor, torch.Tensor, np.ndarray]]): The batch data

            Returns:
                Tuple[torch.Tensor, torch.Tensor, np.ndarray]: The stacked X,y and paths for the given batch
            """
            utils.log_gpus_status()

            # Extract x,y,path from the batch
            Xs, ys, paths = zip(*batch)
            Xs, ys, paths = torch.vstack(Xs), torch.hstack(ys), np.hstack(paths)
            
            # Shuffle x,y, and paths within the batch if needed
            if shuffle:
                indexes = np.arange(len(Xs))
                np.random.shuffle(indexes)
                Xs, ys, paths = Xs[indexes], ys[indexes], paths[indexes]
            
            assert len(Xs) == len(ys) == len(paths)

            logging.info(f"X shape: {Xs.shape}, y shape: {ys.shape}, paths shape: {paths.shape}")

            return Xs, ys, paths
        
        return collate_fn

    @staticmethod
    def get_subset(dataset, indexes:np.ndarray[int]):
        """Get subset of the dataset

        Args:
            dataset (Self): The dataset
            indexes (np.ndarray[int]): The indexes to select from the given dataset

        Returns:
            Self: A subset of the given dataset
        """
        subset = deepcopy(dataset)
        
        subset.X_paths, subset.y = dataset.X_paths[indexes], dataset.y[indexes]
        
        return subset
    
    def set_Xy(self, X_paths:np.ndarray[str], y:np.ndarray[str]):
        """Set the class params

        Args:
            X_paths (np.ndarray[str]): The paths
            y (np.ndarray[str]): The labels
        """
        self.set_X_paths(X_paths)
        self.set_y(y)    
        
        # Store unique labels appears in y
        self.unique_labels = np.unique(self.y)
        
        # Map the labels to ids
        self.__label_to_id_mapping = {label: idx for idx, label in enumerate(self.unique_labels)}
    
    def set_transform(self, transform:transforms.Compose):
        """Set the transform to be applied to the data samples on retrieval

        Args:
            transform (torch.compose): The transform to apply
        """
        self.transform:transforms.Compose = transform
        
    def get_configuration(self)->DatasetConfig:
        """Returns the dataset config

        Returns:
            DatasetConfig: The dataset config
        """
        return self.dataset_config
    
    def get_X_paths(self)->np.ndarray[str]:
        """Get the X paths

        Returns:
            np.ndarray[str]: The X paths
        """
        return self.X_paths
        
    def get_y(self)->np.ndarray[str]:
        """Get the y values

        Returns:
            np.ndarray[str]: y
        """
        return self.y
    
    def set_X_paths(self, new_X_paths:np.ndarray[str]):
        """Set the X paths

        Args:
            new_X_paths (np.ndarray[str]): The values to set
        """
        self.X_paths = new_X_paths
        
    def set_y(self, new_y:np.ndarray[str]):
        """Set the y values

        Args:
            new_y (np.ndarray[str]): The values to set
        """
        self.y = new_y

    
    def id2label(self, ids: Union[int, Iterable[int]]) -> Union[str, np.ndarray[str]]:
        """
        Maps an id or a list of ids to their corresponding labels.

        Args:
            ids (Union[int, Iterable[int]]): A single id or a list/array of ids to be converted.

        Returns:
            Union[str, np.ndarray[str]]: The label or list of labels corresponding to the input ids.
        """
        assert isinstance(ids, int) or isinstance(ids, Iterable), f"ids type ({type(ids)}) isn't supported"

        # If ids is a single integer, return the corresponding label
        if isinstance(ids, int):
            return self.unique_labels[ids]
        
        # If ids is an iterable, return the list of corresponding labels
        return [self.unique_labels[i] for i in ids]
    
    def label2id(self, labels: Union[str, Iterable]) -> Union[int, np.ndarray[int]]:
        """
        Maps a label or an iterable of labels to their corresponding indices.

        Args:
            labels (Union[str, Iterable]): A single label, or an iterable of labels to be converted.

        Returns:
            Union[int, np.ndarray]: The index or list of indices corresponding to the input labels.
        """
        
        assert isinstance(labels, str) or isinstance(labels, Iterable), f"label type ({type(labels)}) isn't supported"
        
        if isinstance(labels, str):
            return self.__label_to_id_mapping[labels]
        
        labels_np = np.asarray(labels)
        return np.vectorize(self.__label_to_id_mapping.get)(labels_np)

    
    def split(self)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits the data into training, validation, and test sets.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Indexes for the training, validation, and test sets.
        """
        
        X_indexes, y = np.arange(len(self.X_paths)), self.y
        train_size = self.dataset_config.TRAIN_PCT
        random_state = self.dataset_config.SEED
        shuffle = self.dataset_config.SHUFFLE
    
        # First, split the data in training and remaining dataset
        logging.info(f"Split data by set (train/val/test): {train_size}")
        train_indexes, temp_indexes = train_test_split(X_indexes, 
                                                        train_size=train_size,
                                                        random_state=random_state,
                                                        shuffle=shuffle,
                                                        stratify=y)

        # The valid and test size to be equal (that is 50% of remaining data)
        val_indexes, test_indexes = train_test_split(temp_indexes,
                                                        test_size=0.5,
                                                        random_state=random_state,
                                                        shuffle=shuffle,
                                                        stratify=y[temp_indexes])
        
        logging.info(f"Train set: {len(train_indexes)}, Validation set: {len(val_indexes)}, Test set: {len(test_indexes)}")
        
        return train_indexes, val_indexes, test_indexes

    @abstractmethod
    def _load_data_paths(self):
        """Abstract method: Load the paths to the data files
        """
        pass
