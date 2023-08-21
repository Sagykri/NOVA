
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
    def __init__(self, conf: DatasetConfig):
        self.__set_params(conf)
        
    def __set_params(self, conf: DatasetConfig):
        self.input_folders = conf.INPUT_FOLDERS
        self.add_condition_to_label = conf.ADD_CONDITION_TO_LABEL
        self.add_line_to_label = conf.ADD_LINE_TO_LABEL
        self.add_batch_to_label = conf.ADD_BATCH_TO_LABEL
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
        
        subset.unique_markers = np.unique(subset.y)
        subset.label = subset.y
        
        return subset
    
    def __getitem__(self, index):
        logging.info(f"\n\n\n\n __getitem__ {index}")
        'Generate one batch of data'
        
        X_batch, y_batch, paths_batch = self.get_batch(index, return_paths=True)
    
        y_batch = self.__label_converter(y_batch, label_format='index')
        paths_batch = paths_batch.reshape(-1,1)
        
        return {'image': X_batch, 'label': y_batch, 'image_path': paths_batch}
        

    def get_batch(self, indexes, return_paths=False):
        if not isinstance(indexes, list):
            indexes = [indexes]
        logging.info(f"Indexes: {indexes}, {self.X_paths[indexes]}")
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
            logging.info(f"Path: {path}")
            imgs = np.load(path)
            
            n_tiles = imgs.shape[0]
            
            augmented_images = []
            
            if self.flip or self.rot:
                for j in range(n_tiles): 
                    # Augmentations
                    img = np.copy(imgs[j]) if not self.is_aug_inplace else imgs[j]
                    if self.flip:
                        img = random_horizontal_flip(img) 
                        img = random_vertical_flip(img) 
                    if self.rot:
                        img = random_choice_rotate(img) 
                    
                    if self.is_aug_inplace:
                        imgs[j] = img
                    elif not np.array_equal(img, imgs[j]): 
                        augmented_images.append(img) 
        
            # Store sample - all the tiles in site
            X_batch.append(imgs)
            y_batch.append([labels[i]]*n_tiles)
            if return_paths:
                paths_batch.append([path]*n_tiles)
            
            if not self.is_aug_inplace:
                # Append augmented images
                if len(augmented_images) > 0: 
                    augmented_images = np.stack(augmented_images) 
                    X_batch.append(augmented_images) 
                    y_batch.append([labels[i]]*len(augmented_images)) 
                    if return_paths:
                        paths_batch.append([path]*len(augmented_images))
        
        X_batch = np.concatenate(X_batch)
        y_batch = np.asarray(utils.flat_list_of_lists(y_batch))
        if return_paths:
            paths_batch = np.asarray(utils.flat_list_of_lists(paths_batch))

        # If the channel axis is the last one, move it to be the second one
        # (#tiles, 100, 100, #channel) -> (#tiles, #channel, 100, 100)
        if np.argmin(X_batch.shape[1:]) == len(X_batch.shape) - 2:
            X_batch = np.moveaxis(X_batch, -1, 1)

        logging.info(f"y_batch shape: {y_batch.shape}")
        
        logging.info(f"\n\n [load_batch]  X_batch: {X_batch.shape}, y [{np.unique(y_batch)}]: {y_batch.shape}")

        
        ###############################################################
        
        res = utils.apply_for_all_gpus(utils.getfreegpumem)
        logging.info(f"Resources (Free, Used, Total): {res}")
        nvidia_smi_info = utils.apply_for_all_gpus(utils.get_nvidia_smi_output)
        logging.info(f"nvidia_smi: {nvidia_smi_info}")
        
        if return_paths:
            return X_batch, y_batch, paths_batch
        
        return X_batch, y_batch
        
    def id2label(self, y_id, unique_markers=None):
        if unique_markers is None:
            unique_markers = self.unique_markers
            
        y_label = np.empty_like(y_id, dtype='object')
        for i in range(len(unique_markers)):
            label = unique_markers[i]
            y_label[y_id==i] = label
            
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
            
            assert output_images.shape[0] == output_labels.shape[0] == output_paths.shape[0] 

            logging.info(f"Image shape: {output_images.shape}, label shape: {output_labels.shape}, label unique: {np.unique(output_labels)}")
            return {'image': output_images, 'label': output_labels, 'image_path': output_paths}
        
        return collate_fn
    
    def split(self):
        """
        Split data by set (train/val/test)
        """
        # Dummy: df = pd.DataFrame([[0, '../A/Site1#2', 'A'], [1, "../A/Site1#3", 'A'], [100, '../B/Site3#4', 'B'], [101, '../B/Site4#6', 'B']], columns=['index', 'path', 'label'])
        # df.loc[:,'paths_sites'] = df['paths'].apply(lambda x: x.split('#')[0])
        # df.groupby('paths_sites').agg({"index": lambda x: list(x), "labels": lambda x: np.unique(x)})
        X_indexes, y = np.arange(len(self.X_paths)), self.y
        train_size = self.train_part
        random_state = self.conf.SEED
        shuffle = self.shuffle
        
        # # For stratification - check train_size is valid
        # n_classes = len(np.unique(y))
        # min_train_size = 1 - n_classes/len(X_indexes)
        # if train_size < min_train_size:
        #     logging.warning(f"'train_size' is lower than allowed. Setting it to be the minimum possible: {min_train_size}")
        #     train_size = min_train_size
    
        # First, split the data in training and remaining dataset
        logging.info(f"Split data by set (train/val/test): {train_size}")
        # logging.info(f"!!! test size is set to 0.3 !!!!")
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
        # Markers order
        train_markers_order = [label[0].split('_')[0] for label in y_train]
        valid_markers_order = [label[0].split('_')[0] for label in y_valid]
        test_markers_order = [label[0].split('_')[0] for label in y_test]
        
        logging.warning("\n\nTODO: Sagy, to implment _labels_changepoints (if needed) + return markers_order!")
        
        # self.train_data, self.train_label, self.train_markers_order = None = X_train, y_train, train_markers_order
        # self.val_data, self.val_label, self.valid_markers_order = X_valid, y_valid, valid_markers_order
        # self.test_data, self.test_label, self.test_markers_order = X_test, y_test, test_markers_order
        
        # self.X_train, self.y_train = X_train, y_train
        # self.X_valid, self.y_valid = X_valid, y_valid
        # self.X_test, self.y_test = X_test, y_test
        # self.unique_markers = unique_markers
        
        # self.train_markers_order = train_markers_order
        # self.valid_markers_order = valid_markers_order
        # self.test_markers_order = test_markers_order
        
        return X_train_indexes, X_valid_indexes, X_test_indexes