
from abc import ABC, ABCMeta, abstractmethod

import logging
import sys
import os
import numpy as np
import torch
from copy import deepcopy

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from sklearn.model_selection import train_test_split
from src.common.configs.dataset_config import DatasetConfig
import src.common.lib.utils as utils
from copy import deepcopy

from collections import defaultdict
##

# TODO: Use torch.transforms instead!
def random_horizontal_flip(image):
    if np.random.rand() < 0.5:
        return np.fliplr(image)
    return image

def random_vertical_flip(image):
    if np.random.rand() < 0.5:
        return np.flipud(image)
    return image

def random_choice_rotate(image):
    k = np.random.choice([0, 1, 2, 3])
    return np.rot90(image, k=k, axes=(0,1))


class Dataset(torch.utils.data.Dataset ,metaclass=ABCMeta):
    def __init__(self, conf: DatasetConfig, transform=None):
        self.__set_params(conf)
        self.transform = transform
        
    def __set_params(self, conf: DatasetConfig):
        self.input_folders = conf.INPUT_FOLDERS
        self.add_condition_to_label = conf.ADD_CONDITION_TO_LABEL
        self.add_line_to_label = conf.ADD_LINE_TO_LABEL
        self.add_batch_to_label = conf.ADD_BATCH_TO_LABEL
        self.add_rep_to_label   = conf.ADD_REP_TO_LABEL
        self.add_type_to_label = conf.ADD_TYPE_TO_LABEL
        self.markers = conf.MARKERS
        self.markers_to_exclude = conf.MARKERS_TO_EXCLUDE
        self.cell_lines = conf.CELL_LINES
        self.conditions = conf.CONDITIONS
        self.train_part = conf.TRAIN_PCT
        self.shuffle = conf.SHUFFLE
        self.flip = conf.AUG_TO_FLIP
        self.rot = conf.AUG_TO_ROT
        self.to_split_data = conf.SPLIT_DATA
        self.data_set_type = conf.DATA_SET_TYPE
        self.is_aug_inplace = conf.IS_AUG_INPLACE
        self.reps = conf.REPS
        

        self.conf = conf
        
        self.X_paths, self.y, self.unique_markers = self._load_data_paths()  
        
        # PATCH...
        self.label = self.y      
        
    @abstractmethod
    def _load_data_paths(self):
        pass
    
    """
    # TODO: (take from improvement/preprocessing)
    def get_variance():
        from src.common.lib.image_sampling_utils import sample_images_all_markers_all_lines
    
        paths = sample_images_all_markers_all_lines(50)
        
        images = np.concatenate([np.load(path) for path in paths])
        
        return np.var(images)
    """
    
    def __len__(self):
        return len(self.y)
    
    @staticmethod
    def get_subset(dataset, indexes):
        subset = deepcopy(dataset)
        
        subset.X_paths, subset.y = dataset.X_paths[indexes], dataset.y[indexes]
        
        subset.label = subset.y
        
        return subset
    
    # Function to apply the transform to each sample in the batch
    def __apply_transform_per_sample(self, batch, transform):
        batch_size = batch.size(0)
        transformed_batch = []
        for i in range(batch_size):
            transformed_batch.append(transform(batch[[i]]))
        transformed_batch = torch.vstack(transformed_batch)
        return transformed_batch
    
    def __apply_paired_transform_batch(self, batch, transform):
        batch_size = batch.size(0)
        batch_global = torch.empty_like(batch)
        batch_local = torch.empty_like(batch)
        
        # Apply transform in parallel
        for i in range(batch_size):
            img_global, img_local = transform(batch[[i]])
            batch_global[i] = img_global
            batch_local[i] = img_local
            
        return batch_global, batch_local
    
    def __getitem__(self, index):
        'Generate one batch of data'
        X_batch, y_batch, paths_batch = self.get_batch(index, return_paths=True)
    
        y_batch = self.__label_converter(y_batch, label_format='index')
        paths_batch = paths_batch.reshape(-1,1)
        
        is_dual_inputs = utils.get_if_exists(self.conf, 'IS_DUAL_INPUTS', False)
        if is_dual_inputs:
            if self.transform:
                
                X_batch = torch.from_numpy(X_batch)  
                X_batch_global, X_batch_local = self.__apply_paired_transform_batch(X_batch, self.transform)
                
                return {'image_global': X_batch_global, 'image_local': X_batch_local, 'label': y_batch, 'image_path': paths_batch}        

        
        if self.transform:
            X_batch = torch.from_numpy(X_batch)
            X_batch = self.__apply_transform_per_sample(X_batch, self.transform)
            
        return {'image': X_batch, 'label': y_batch, 'image_path': paths_batch}        

    def get_batch(self, indexes, return_paths=False):
        if not isinstance(indexes, list):
            indexes = [indexes]
        X_paths_batch = self.X_paths[indexes]
        y_batch = self.y[indexes]

        return self.__load_batch(X_paths_batch, y_batch, return_paths=return_paths)

    def __load_batch(self, paths, labels, return_paths=False):
        'Generates data containing batch_size samples' 
        X_batch = []
        y_batch = []
        paths_batch = []
        
        # Generate data
        for i, path in enumerate(paths):
            imgs = np.load(path)
            
            n_tiles = imgs.shape[0]
        
            # Store sample - all the tiles in site
            X_batch.append(imgs)
            y_batch.append([labels[i]]*n_tiles)
            if return_paths:
                paths_batch.append([path]*n_tiles)
            
   
        X_batch = np.concatenate(X_batch)
        y_batch = np.asarray(utils.flat_list_of_lists(y_batch))
        if return_paths:
            paths_batch = np.asarray(utils.flat_list_of_lists(paths_batch))

        # If the channel axis is the last one, move it to be the second one
        # (#tiles, 100, 100, #channel) -> (#tiles, #channel, 100, 100)
        if np.argmin(X_batch.shape[1:]) == len(X_batch.shape) - 2:
            X_batch = np.moveaxis(X_batch, -1, 1)
        
        if return_paths:
            return X_batch, y_batch, paths_batch
        
        return X_batch, y_batch
        
    def id2label(self, y_id):
        y_label = self.unique_markers[y_id.flatten().astype(int)]
        
        return y_label
    
    def __label_converter(self, y, label_format='index'):
        if self.unique_markers is None:
            raise ValueError('unique_markers is empty.')
        else:
            y = y.reshape(-1,)
            onehot = y[:, None] == self.unique_markers
            if label_format == 'onehot':
                output = onehot
            elif label_format == 'index':
                output = onehot.argmax(1)
            else:
                output = y

            output = output.reshape(-1,1)
            return output
    
    @staticmethod
    def get_collate_fn(shuffle=False):
        def collate_fn(batch):
            res = utils.apply_for_all_gpus(utils.getfreegpumem)
            logging.info(f"Resources (Free, Used, Total): {res}")

            images = [b['image'] for b in batch]
            labels = [b['label'] for b in batch]
            images_paths = [b['image_path'] for b in batch]
            
            output_images = torch.from_numpy(np.vstack(images))
            output_labels = torch.from_numpy(np.vstack(labels).reshape(-1,))
            output_paths = np.vstack(images_paths).reshape(-1,)
            
            if shuffle:
                indexes = np.arange(len(output_images))
                np.random.shuffle(indexes)
                output_images = output_images[indexes]
                output_labels = output_labels[indexes]
                output_paths = output_paths[indexes]
            
            assert output_images.shape[0] == output_labels.shape[0]

            logging.info(f"Image shape: {output_images.shape}, label shape: {output_labels.shape}")

            return {'image': output_images, 'label': output_labels, 'image_path': output_paths}
        
        return collate_fn
    
    def split(self):
        """
        Split data by set (train/val/test)
        """
        X_indexes, y = np.arange(len(self.X_paths)), self.y
        train_size = self.train_part
        random_state = self.conf.SEED
        shuffle = self.shuffle
    
        # First, split the data in training and remaining dataset
        logging.info(f"Split data by set (train/val/test): {train_size}")
        X_train_indexes, X_temp_indexes, y_train, y_temp = train_test_split(X_indexes, y,
                                                                            train_size=train_size,
                                                                            random_state=random_state,
                                                                            shuffle=shuffle,
                                                                            stratify=y)

        # The valid and test size to be equal (that is 50% of remaining data)
        X_valid_indexes, X_test_indexes, y_valid, y_test = train_test_split(X_temp_indexes, y_temp,
                                                                            test_size=0.5,
                                                                            random_state=random_state,
                                                                            shuffle=shuffle,
                                                                            stratify=y_temp)
        
        logging.info(f"Train set: {len(X_train_indexes)} {len(y_train)}")
        logging.info(f"Validation set: {len(X_valid_indexes)} {len(y_valid)}")
        logging.info(f"Test set: {len(X_test_indexes)} {len(y_test)}")
        
        return X_train_indexes, X_valid_indexes, X_test_indexes
