
from abc import ABC, abstractmethod
import logging
import sys
import os
import numpy as np

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from sklearn.model_selection import train_test_split

from src.common.configs.dataset_config import DatasetConfig



class Dataset(ABC):
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
        
        self.conf = conf
        
        self.X_paths, self.y, self.unique_markers = self._load_data_paths()        
        
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